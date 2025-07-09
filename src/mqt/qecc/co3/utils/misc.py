"""Misc functions for plotting and Benchmarking."""

from __future__ import annotations

import math
import random
import warnings
from typing import TYPE_CHECKING

import numpy as np
import qiskit as qk
from qiskit.quantum_info import random_statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

if TYPE_CHECKING:
    import mqt.qecc.co3 as co

random.seed(45)


def generate_max_parallel_circuit(q: int, min_depth: int) -> list[tuple[int, int] | int]: # actually only tuples but mypy needs the int elements too
    """Circuits with maximally parallelizable layers, i.e. per layer, ALL qubits are used in disjoint gates.

    CNOTS only.
    To make it less arbitrary, you should choose min depth to be a multiple of q, i.e. s*q, s.t. you get 2s layers
    Otherwise, the last layer might be a bit empty.
    """
    gates_counter = 0
    circuit: list[tuple[int, int] | int] = []
    labels = list(range(q))
    while gates_counter <= min_depth:
        random.shuffle(labels)
        tuples = [(labels[i], labels[i + 1]) for i in range(0, len(labels), 2)]
        gates_counter += len(tuples)
        circuit += tuples
        if gates_counter == min_depth:
            break

    return circuit


def generate_min_parallel_circuit(q: int, min_depth: int, layer_size: int) -> list[tuple[int, int] | int]: #same as above
    """Circuits which have nearly no parallelism at all.

    CNOTS only.
    One could enforce that each consecutive gate shares one qubit with the one before, but then there would be
    NO parallelism at all and then, the hc and routing would trivially have no benefit and no parallelism.
    Hence, choose a layer_size, maybe 2 or 3 which ensures that there are max. 2 or 3 gates per layer until a qubit is shared again.
    """
    num_layers = min_depth // layer_size
    lst = []
    all_labels_used = set()  # Track which labels have been used

    # first layer
    labels = list(range(q))
    random.shuffle(labels)
    tuples = [(labels[i], labels[i + 1]) for i in range(0, len(labels), 2)]
    first_layer = random.sample(tuples, layer_size)
    lst.append(first_layer)

    all_labels_used.update([label for tup in first_layer for label in tup])

    # gen subsequent layers s.t. at least one qubit label overlaps.
    while len(all_labels_used) < q or len(lst) < num_layers:
        temp = []
        flattened_labels = [label for tup in lst[-1] for label in tup]
        k = random.choice(  # noqa: S311
            flattened_labels
        )  # this qubit will be used in current layer too to destroy parallelism
        labels_copy = labels.copy()
        labels_copy.remove(k)
        l = random.choice(labels_copy)  # form pair with l and k  # noqa: E741, S311
        temp.append((l, k))
        labels_copy.remove(l)

        # fill up layer, avoid duplicates
        while len(temp) < layer_size:
            random_tuple = random.choice([(labels_copy[i], labels_copy[i + 1]) for i in range(0, len(labels_copy), 2)])  # noqa: S311
            # Check for duplicates in the layer
            if all(
                t[0] not in [tup[0] for tup in temp] and t[1] not in [tup[1] for tup in temp] for t in [random_tuple]
            ):
                temp.append(random_tuple)
                labels_copy.remove(random_tuple[0])
                labels_copy.remove(random_tuple[1])

        all_labels_used.update([label for tup in temp for label in tup])
        lst.append(temp)

    # flatten tuples
    circuit: list[tuple[int, int] | int] = []
    for el in lst:
        circuit += el

    return circuit


def generate_random_circuit(
    q: int, min_depth: int, tgate: bool = False, ratio: float = 0.5
) -> list[tuple[int, int] | int]:
    """Random CNOT Pairs. Optional: random T gates.

    makes it deep enough that each qubit is used at least once
    min_depth is the minimum number of cnots
    circuit = set of terminal pairs
    the labeling does not yet follow the labels of a networkx.Graph but only range(q).

    Note: min_depth should in principle be larger than q.

    Args:
        q (int): number of qubits of the circuit
        min_depth (int): minimal number of gates
        tgate (bool, optional): whether t gates are included or not Defaults to False.
        ratio (float, optional): ratio between t gates and cnots.
            more t gates if smaller than 0.5.
            note that the ratio is not deterministically fixed, only determines probabilities.
            Defaults to 0.5.
            ratio = num_cnots/(num_t + num_cnots)

    Raises:
        ValueError: _description_

    Returns:
        list[tuple[int, int]]: _description_
    """
    if q < 2:
        msg = "q must be at least 2 to form pairs."
        raise ValueError(msg)

    # predetermine the desired number of t gates and cnots
    num_cnot_gates = round(min_depth * ratio) if tgate else min_depth
    num_t_gates = min_depth - num_cnot_gates

    cnot_pairs: list[tuple[int, int]] = []
    t_gates: list[int] = []
    used_qubits = set()

    # Ensure each qubit is used at least once
    available_qubits = list(range(q))
    random.shuffle(available_qubits)

    while len(cnot_pairs) <= num_cnot_gates:
        a, b = random.sample(range(q), 2)
        cnot_pairs.append((a, b))
        used_qubits.update([a, b])

    if tgate is True:
        while len(t_gates) <= num_t_gates:
            a = random.randrange(q)  # noqa: S311
            t_gates.append(a)
            used_qubits.add(a)

    # check whether qubit labels are unused and if yes, add gates in accordance to ratio
    missing_qubits = set(range(q)) - used_qubits
    extra_cnot_count = num_cnot_gates
    extra_t_count = num_t_gates

    for i in missing_qubits:
        # Compute current ratio dynamically
        total_gates = extra_cnot_count + extra_t_count
        expected_cnot_count = round(total_gates * ratio) if tgate else total_gates
        expected_t_count = total_gates - expected_cnot_count

        if extra_t_count < expected_t_count:
            t_gates.append(i)  # Prioritize adding T gate
            extra_t_count += 1
        else:
            b = random.choice(range(q))  # Pick a random second qubit  # noqa: S311
            while b == i:  # Ensure b is different from i
                b = random.choice(range(q))  # noqa: S311
            cnot_pairs.append((i, b))
            extra_cnot_count += 1

    circuit = list(cnot_pairs) + list(t_gates)
    num_c = len(list(cnot_pairs))
    num_t = len(list(t_gates))
    final_ratio = num_c / (num_c + num_t)
    assert abs(ratio - final_ratio) < 0.07, (
        "The final ratio deviates more than 0.05 from desired ratio= cnot/total gates"
    )
    random.shuffle(circuit)

    return circuit


def translate_layout_circuit(
    pairs: list[tuple[int, int] | int], layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]]
) -> list[tuple[tuple[int, int], tuple[int, int]] | tuple[int, int]]:
    """Translates a `pairs` circuit (with int labels) into the lattice's labels for a given layout.

    However, pairs does not only include tuple[int,int] but can include int as well for T gates. Then, layout will also include
    a list of factory positions in the key="factory_positions". but this will be ignored for this
    """
    # return [(layout[pair[0]], layout[pair[1]]) for pair in pairs]
    # terminal_pairs = [(layout[pair[0]], layout[pair[1]]) if isinstance(pair, tuple) else layout[pair] for pair in pairs]
    terminal_pairs: list[tuple[tuple[int, int], tuple[int, int]] | tuple[int, int]] = []
    for pair in pairs:
        if isinstance(pair, tuple):
            pos1_raw = layout[pair[0]]
            pos2_raw = layout[pair[1]]
            if not isinstance(pos1_raw, tuple) or not isinstance(pos2_raw, tuple):
                msg = "Expected tuple[int, int] in layout mapping."
                raise TypeError(msg)
            pos1 = (int(pos1_raw[0]), int(pos1_raw[1]))
            pos2 = (int(pos2_raw[0]), int(pos2_raw[1]))
            terminal_pairs.append((pos1, pos2))
        else:
            pos_raw = layout[pair]
            if not isinstance(pos_raw, tuple):
                msg = "Expected tuple[int, int] in layout mapping."
                raise TypeError(msg)
            pos = (int(pos_raw[0]), int(pos_raw[1]))
            terminal_pairs.append(pos)

    return terminal_pairs


def compare_original_dynamic_gate_order(
    q: int, layout: dict[int | str, tuple[int, int] | list[tuple[int, int]]], router: co.ShortestFirstRouterTGatesDyn
) -> bool:
    """Generates a qiskit circuit for both the order after doing dynamic routing and the original order.

    Hence, it is checked whether the many reorderings in dynamic routing are really safe and sound.

    Args:
        q (int): number of qubits
        router (co.ShortestFirstRouterTGatesDyn): router to be checked
        layout (dict): must be the same layout with which the router's terminal_pairs were initialized.

    Returns:
        bool: Whether the final states coincide, i.e. whether dynamic routing is safe.
    """
    assert q <= 20, (
        "Too many qubits cannot be simulated via qiskit statevector simulator anymore. Consder less than 20 qubits."
    )

    # run the dynamic routing once
    vdp_layers_dyn = router.find_total_vdp_layers_dyn()

    # original layers
    gates_previous = []
    for lst in router.layers_copy:
        gates_previous += lst

    # gate order after routing
    gates_routing = []
    for dct in vdp_layers_dyn:
        gates_routing += list(dct.keys())

    # warning if both lists are identical. Then, the test is trivial
    if gates_routing == gates_previous:
        warnings.warn(
            "The test of comparing initial and post-dyn-routing order is trivial if both gates are ordered the same way. Try again to sample a new random circuit.",
            category=RuntimeWarning,
            stacklevel=2,
        )

    reverse_mapping = {v: k for k, v in layout.items()}
    translated_previous: list[tuple[int | str, ...] | int | str] = []
    for item in gates_previous:
        if isinstance(item, tuple):  # If it's a tuple, check if it's a nested pair
            if isinstance(item[0], tuple):  # If it's a tuple of tuples (nested)
                translated_previous.append(tuple(reverse_mapping[sub] for sub in item))
            else:  # If it's a single tuple directly in the list
                translated_previous.append(reverse_mapping[item])
        # else:  # If it's not a tuple (single number, shouldn't happen based on your input)
        #    msg = f"Unexpected element in data: {item}"
        #    raise TypeError(msg)

    translated_routing: list[tuple[int | str, ...] | int | str] = []
    for item in gates_routing:
        if isinstance(item, tuple):  # If it's a tuple, check if it's a nested pair
            if isinstance(item[0], tuple):  # If it's a tuple of tuples (nested)
                translated_routing.append(tuple(reverse_mapping[sub] for sub in item))
            else:  # If it's a single tuple directly in the list
                translated_routing.append(reverse_mapping[item])
        # else:  # If it's not a tuple (single number, shouldn't happen based on your input)
        #    msg = f"Unexpected element in data: {item}"
        #    raise TypeError(msg)

    # switch on purpose two entries which is wrong to check whether this is recognized
    # temp = translated_routing[0]
    # translated_routing[0] = translated_routing[-1]
    # translated_routing[-1] = temp

    # initialize random state (s.t. CNOT and T are not trivially appplied)
    random_state = random_statevector(2**q)

    # build circuits for previous and after dyn routing order
    # original input order
    qc_previous = qk.QuantumCircuit(q)
    qc_previous.initialize(random_state, range(q))
    qc_previous.barrier()

    for op in translated_previous:
        if isinstance(op, tuple):  # Apply CNOT for (control, target)
            qc_previous.cx(op[0], op[1])
        else:  # Apply Hadamard for single qubit
            qc_previous.h(op)

    backend = AerSimulator(method="statevector")
    qc_previous.save_statevector()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    qc_combine = pm.run(qc_previous)

    result = backend.run(qc_combine, shots=1)
    psi_out_complex = result.result()

    # after dyn routing order
    qc_routing = qk.QuantumCircuit(q)
    qc_routing.initialize(random_state, range(q))
    qc_routing.barrier()

    for op in translated_routing:
        if isinstance(op, tuple):  # Apply CNOT for (control, target)
            qc_routing.cx(op[0], op[1])
        else:  # Apply Hadamard for single qubit
            qc_routing.h(op)

    backend2 = AerSimulator(method="statevector")
    qc_routing.save_statevector()
    pm2 = generate_preset_pass_manager(backend=backend2, optimization_level=1)
    qc_combine = pm2.run(qc_routing)

    result2 = backend2.run(qc_routing, shots=1)
    psi_out_complex_2 = result2.result()

    diff = np.linalg.norm(psi_out_complex.data()["statevector"] - psi_out_complex_2.data()["statevector"])
    return math.isclose(diff, 0, abs_tol=1e-14)

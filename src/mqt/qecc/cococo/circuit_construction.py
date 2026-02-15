# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Misc functions for plotting and Benchmarking."""

from __future__ import annotations

import random

from qiskit import QuantumRegister
from qiskit.circuit.library import CXGate
from qiskit.dagcircuit import DAGCircuit


def create_random_sequential_circuit_dag(
    j: int, q: int, num_gates: int, seed: int = 45
) -> tuple[DAGCircuit, list[tuple[int, int]]]:
    """Creates a sequential circuit with j gates per layer on q qubits with at least num_gates in total.

    takes layers from perspective of DAG into account
    """
    rng = random.Random(seed)  # noqa: S311

    dag = DAGCircuit()
    qreg = QuantumRegister(q, "q")
    dag.add_qreg(qreg)

    pairs = []  # final return
    layers_int = []

    num_layers = (num_gates + j - 1) // j

    if q // 2 < j:
        msg = "j is too large, cannot fit that many disjoint gates in a layer with q qubits"
        raise ValueError(msg)

    # first layer fully random
    layer_int_temp: list[tuple[int, int]] = []
    for _l in range(j):
        # sample control and target randomly
        flat = [x for pair in layer_int_temp for x in pair]
        while True:
            c = rng.randint(0, q - 1)
            if c not in flat:
                break
        while True:
            t = rng.randint(0, q - 1)
            if c != t and t not in flat:
                break
        layer_int_temp.append((c, t))
        dag.apply_operation_back(CXGate(), qargs=[qreg[c], qreg[t]])

    pairs += layer_int_temp
    layers_int.append(layer_int_temp)

    # choose random qubits of the first layer
    temp_reuse = []
    temp_noreuse = []
    for pair in layer_int_temp:
        chosen = random.choice(pair)  # noqa: S311
        temp_reuse.append(chosen)
        other = pair[0] if chosen == pair[1] else pair[1]
        temp_noreuse.append(other)

    # next layers are done based on the previous layer
    for _i in range(1, num_layers):
        layer_int_temp = []

        temp_reuse_second = []
        temp_noreuse_second = []
        # go through these temp_reuse gates and sample one other qubit and create random c,t order
        for qubit in temp_reuse:
            flat = [
                x for pair in layer_int_temp for x in pair
            ]  # make sure that qubits are not reused in the same layer
            while True:
                qubit2 = rng.randint(0, q - 1)
                if qubit != qubit2 and qubit2 not in flat and qubit2 in temp_noreuse:
                    break
            temp_reuse_second.append(qubit2)
            temp_noreuse_second.append(qubit)
            temp_lst = [qubit, qubit2]
            rng.shuffle(temp_lst)
            # print("temp lst", temp_lst)
            layer_int_temp.append((temp_lst[0], temp_lst[1]))
            dag.apply_operation_back(CXGate(), qargs=[qreg[temp_lst[0]], qreg[temp_lst[1]]])
        layers_int.append(layer_int_temp)
        pairs += layer_int_temp
        temp_reuse = temp_reuse_second
        temp_noreuse = temp_noreuse_second

    return dag, pairs


def generate_max_parallel_circuit(
    q: int, min_depth: int, seed: int = 45
) -> list[tuple[int, int] | int]:  # actually only tuples but mypy needs the int elements too
    """Circuits with maximally parallelizable layers, i.e. per layer, ALL qubits are used in disjoint gates.

    CNOTS only.
    To make it less arbitrary, you should choose min depth to be a multiple of q, i.e. s*q, s.t. you get 2s layers
    Otherwise, the last layer might be a bit empty.
    """
    rng = random.Random(seed)  # noqa: S311

    gates_counter = 0
    if q < 2 or q % 2 != 0:
        msg = "q must be an even integer larger than 2."
        raise ValueError(msg)
    circuit: list[tuple[int, int] | int] = []
    labels = list(range(q))
    while gates_counter <= min_depth:
        rng.shuffle(labels)
        tuples = [(labels[i], labels[i + 1]) for i in range(0, len(labels), 2)]
        gates_counter += len(tuples)
        circuit += tuples
        if gates_counter == min_depth:
            break

    return circuit


def generate_min_parallel_circuit(
    q: int, min_depth: int, layer_size: int, seed: int = 45
) -> list[tuple[int, int] | int]:  # same as above
    """Circuits which have nearly no parallelism at all.

    CNOTS only.
    One could enforce that each consecutive gate shares one qubit with the one before, but then there would be
    NO parallelism at all and then, the hc and routing would trivially have no benefit and no parallelism.
    Hence, choose a layer_size, maybe 2 or 3 which ensures that there are max. 2 or 3 gates per layer until a qubit is shared again.
    """
    rng = random.Random(seed)  # noqa: S311

    num_layers = min_depth // layer_size
    lst = []
    all_labels_used = set()  # Track which labels have been used

    if q < 2 or q % 2 != 0:
        msg = "q must be an even integer larger than 2."
        raise ValueError(msg)
    max_layer_size = q // 2
    if not (1 <= layer_size <= max_layer_size):
        msg = "layer_size must be larger than 1 and lower than q//2."
        raise ValueError(msg)

    # first layer
    labels = list(range(q))
    rng.shuffle(labels)
    tuples = [(labels[i], labels[i + 1]) for i in range(0, len(labels), 2)]
    first_layer = rng.sample(tuples, layer_size)
    lst.append(first_layer)

    all_labels_used.update([label for tup in first_layer for label in tup])

    # gen subsequent layers s.t. at least one qubit label overlaps.
    while len(all_labels_used) < q or len(lst) < num_layers:
        temp = []
        flattened_labels = [label for tup in lst[-1] for label in tup]
        k = rng.choice(flattened_labels)  # this qubit will be used in current layer too to destroy parallelism
        labels_copy = labels.copy()
        labels_copy.remove(k)
        l = rng.choice(labels_copy)  # form pair with l and k  # noqa: E741
        temp.append((l, k))
        labels_copy.remove(l)

        # fill up layer, avoid duplicates
        while len(temp) < layer_size:
            pairs = [(labels_copy[i], labels_copy[i + 1]) for i in range(0, len(labels_copy) - 1, 2)]
            if not pairs:
                break
            random_tuple = rng.choice(pairs)
            used = {v for p in temp for v in p}
            if random_tuple[0] not in used and random_tuple[1] not in used:
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
    q: int, min_depth: int, tgate: bool = False, ratio: float = 0.5, seed: int = 45
) -> list[tuple[int, int] | int]:
    """Random CNOT Pairs. Optional: random T gates.

    makes it deep enough that each qubit is used at least once
    min_depth is the minimum number of gates
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
        seed (int): seed for generation

    Returns:
        list[tuple[int, int]]: random circuit of cnot gates and t gates.
    """
    rng = random.Random(seed)  # noqa: S311

    if q < 2:
        msg = "q must be at least 2 to form pairs."
        raise ValueError(msg)

    # predetermine the desired number of t gates and cnots
    num_cnot_gates = round(min_depth * ratio) if tgate else min_depth
    num_t_gates = min_depth - num_cnot_gates

    cnot_pairs: list[tuple[int, int]] = []
    t_gates: list[int] = []
    used_qubits = set()

    while len(cnot_pairs) < num_cnot_gates:
        a, b = rng.sample(range(q), 2)
        cnot_pairs.append((a, b))
        used_qubits.update([a, b])

    if tgate:
        while len(t_gates) < num_t_gates:
            a = rng.randrange(q)
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
            b = rng.choice(range(q))  # Pick a random second qubit
            while b == i:  # Ensure b is different from i
                b = rng.choice(range(q))
            cnot_pairs.append((i, b))
            extra_cnot_count += 1

    circuit = list(cnot_pairs) + list(t_gates)
    num_c = len(list(cnot_pairs))
    num_t = len(list(t_gates))
    final_ratio = num_c / (num_c + num_t)
    if tgate:
        assert abs(ratio - final_ratio) < 0.07, (
            "The final ratio deviates more than 0.07 from desired ratio= cnot/total gates"
        )
    rng.shuffle(circuit)

    return circuit

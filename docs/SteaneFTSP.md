---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Steane-type Fault-Tolerant State Preparation

This module provides automated synthesis for **Steane-type Fault-Tolerant State Preparation (FTSP)** circuits. This approach is particularly well-suited for quantum computing architectures that support a high degree of gate-level parallelism, such as trapped-ion and neutral atom quantum computers.

Unlike other methods that rely on measuring stabilizers directly on the data qubits (flag-based or standard syndrome extraction), Steane-type preparation works by initializing separate logical ancilla states and interacting them with the data state via transversal CNOT gates to detect errors.

## The Challenge: Error Cancellation

The core principle of Steane-type error detection is to copy errors from a data block to an ancilla block using transversal CNOTs and then measure the ancilla to detect the error.

However, if we consider more than just one error to occur under circuit level noise models, so errors can also happen on the ancilla qubits, then this approach can lead to a critical issue: **Error Cancellation**.
For example, if the data state and the ancilla state are prepared using the _exact same_ circuit structure, they suffer from identical propagated errors at identical locations error locations.

When two identical errors meet at a transversal CNOT gate, they can cancel each other out on the target (ancilla) while remaining on the control (data). For example, if an X-error occurs on the $i$-th qubit of both the data and ancilla, the transversal CNOT interaction may result in no detectable syndrome on the ancilla. Consequently, the error on the data block goes undetected, breaking fault tolerance.

### Solution: Structurally Distinct Circuits

To ensure fault tolerance, we must break this symmetry. We need to prepare logical states using circuits that are **logically equivalent** (they prepare the same quantum state) but **structurally distinct** regarding error propagation.
In other words, we want to synthesize circuits in which errors propagate differently, so that no matter where an error occurs, we still get a detectable syndrome on the ancilla.

#### Example: The Steane Code ($d=3$)

To see all of the above in action, let's consider the Steane code, which is a $[[7,1,3]]$ CSS code.
Since the code distance is $d=3$, it can correct up to $t=1$ error. Therefore, we only focus on single errors in the circuit and how they propagate.
We can find a state preparation circuit for the logical $|0\rangle_L$ state using the QECC synthesis tools. This will be our reference circuit, and we can analyze its fault set to understand how errors propagate through it.

```{code-cell} ipython3
from mqt.qecc import CSSCode

steane_code = CSSCode.from_code_name("Steane")
print(steane_code.stabs_as_pauli_strings())
```

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis.state_prep import heuristic_prep_circuit

steane_circ = heuristic_prep_circuit(steane_code)
steane_circ.circ.draw('mpl')
```

Errors propagate this circuit in a specific way, which we can analyze by extracting the fault set.
The fault set lists propagated errors which are the result of a errors within the circuit.
For example, a single X-error on $q_1$ just before the last CNOT gate will propagate to a weight two error on qubits $q_1$ and $q_5$.
Looking at the fault set of the circuit above, we can see this error listed here.

```{code-cell} ipython3
steane_circ.compute_fault_sets()
steane_circ.x_fault_sets[0].faults
```

Note that the fault set is reduced and ignores stabilizer-equivalent errors. Meaning that the propagated error on $q_0$ and $q_4$ is not listed, since it is equivalent to the error on $q_1$ and $q_5$ up to multiplication by the first stabilizer generator $XXIIXXI$.

The bottom line is that the fault set above characterizes the error propagation of the given circuit. So if we were to synthesize a second circuit for the same logical state, that has a different fault set, we can be sure that no errors $E$ of weight up to $\text{wt}(E)=1$ can cancel each other out.

For this, QECC provides a function called `heuristic_reference_prep_circuit` which takes a reference circuit and its fault set as input and tries to find a new circuit that avoids the same faults.

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis.state_prep import heuristic_reference_prep_circuit

steane_circ_alt = heuristic_reference_prep_circuit(steane_code, ref_x_fs=steane_circ.x_fault_sets[0].faults)
steane_circ_alt.circ.draw('mpl')

```

```{code-cell} ipython3
steane_circ_alt.compute_fault_sets()
steane_circ_alt.x_fault_sets[0].faults
```

The two fault sets are distinct also under multiplication by stabilizers. One can verify this by multiplying each of the propagated errors with the $X$ stabilizer generators of the Steane Code and ensuring that errors are not equivalent.

## The 4-Circuit Protocol (Distance $\ge 5$)

For higher-distance codes ($d \geq 5$), simply avoiding identical errors is not enough. We must ensure that the verification process itself does not introduce high-weight errors that go undetected.

This module implements a constant-overhead protocol that uses **four specific circuits** to guarantee strict fault tolerance. This protocol is general and works even for codes without symmetries, such as the $[[17,1,5]]$ color code.

### The Protocol Structure

The protocol requires synthesizing four unique circuits ($C_1, C_2, C_3, C_4$) that interact in a specific sequence:

1.  **$C_1$ (Data Candidate):** The initial state we wish to prepare.
2.  **$C_2$ (X-Verifier):** An ancilla used to detect X-errors on $C_1$.
    - _Constraint:_ Must have a distinct X-fault set from $C_1$.
3.  **$C_3$ (Z-Verifier):** An ancilla used to detect Z-errors on $C_1$.
    - _Constraint:_ Must have a distinct Z-fault set from _both_ $C_1$ and $C_2$ (since Z-errors can propagate backward from ancilla to data).
4.  **$C_4$ (Verifier of the Verifier):** An ancilla used to detect X-errors on $C_3$.
    - _Constraint:_ Ensures that the Z-verification step ($C_3$) did not introduce undetectable X-errors. It must be distinct from the relevant faults of the previous chains.

### Utility: Automated 4-Circuit Synthesis Wrapper

While you can call the synthesis API manually for each step, you can use the following custom helper function in your workflow to automatically generate the full suite of 4 mutually distinct circuits based on the protocol rules.

However, note that for codes with higher distance, the search space for circuits grows significantly, and hence the synthesis process might get stuck sometimes.
Therefore, this utility function is not guaranteed to always find a solution and might be more of a starting point for your own custom synthesis workflow.

```{code-cell} ipython3
import numpy as np
from mqt.qecc.circuit_synthesis import heuristic_prep_circuit
from mqt.qecc.circuit_synthesis.synthesis_utils import check_mutually_disjointness_spcs, get_fs_based_on_d

def synthesize_four_ft_circuits(c1_circuit):
    """
    Synthesizes the four distinct state preparation circuits required
    for Steane-type Fault-Tolerant State Preparation.

    Args:
        c1_circuit: The reference state preparation circuit (C1).

    Returns:
        List of 4 circuits [C1, C2, C3, C4] satisfying t-distinctness constraints.
    """
    # ---------------------------------------------------------
    # Step 1: Analyze Reference Circuit (C1)
    # ---------------------------------------------------------
    c1_x_faults, c1_z_faults = get_fs_based_on_d(c1_circuit)

    # ---------------------------------------------------------
    # Step 2: Synthesize X-Verifier (C2)
    # Constraint: X-faults must be distinct from C1.
    # ---------------------------------------------------------
    c2_circuit = heuristic_reference_prep_circuit(c1_circuit.code, ref_x_fs=c1_x_faults)
    #BUG: Distance is not properly initialized
    c2_circuit.code.distance = c1_circuit.code.distance
    c2_x_faults, c2_z_faults = get_fs_based_on_d(c2_circuit)

    # ---------------------------------------------------------
    # Step 3: Synthesize Z-Verifier (C3)
    # Constraint: Z-faults must be distinct from BOTH C1 and C2.
    # ---------------------------------------------------------
    # Combine Z-faults from C1 and C2 to avoid them simultaneously
    if c1_z_faults.size and c2_z_faults.size:
        combined_z_faults = np.unique(np.vstack((c1_z_faults, c2_z_faults), dtype=np.int8), axis=0)
    elif c1_z_faults.size:
        combined_z_faults = c1_z_faults
    elif c2_z_faults.size:
        combined_z_faults = c2_z_faults
    else:
        # If no Z-faults exist that require distinctness, fallback to reusing circuits
        return [c1_circuit, c2_circuit, c1_circuit, c2_circuit]

    c3_circuit = heuristic_reference_prep_circuit(c1_circuit.code, ref_z_fs=combined_z_faults, guide_by_x=False)
    #BUG: Distance is not properly initialized
    c3_circuit.code.distance = c1_circuit.code.distance
    c3_x_faults, c3_z_faults = get_fs_based_on_d(c3_circuit)

    # ---------------------------------------------------------
    # Step 4: Synthesize Final Verifier (C4)
    # Constraint: X-faults distinct from C3, Z-faults distinct from C1 & C2.
    # ---------------------------------------------------------
    c4_circuit = heuristic_reference_prep_circuit(
        c1_circuit.code,
        ref_x_fs=c3_x_faults,
        ref_z_fs=combined_z_faults,
        guide_by_x=False
    )

    return [c1_circuit, c2_circuit, c3_circuit, c4_circuit]
```

### Automated Synthesis with the $[[17,1,5]]$ Color Code

The $[[17,1,5]]$ color code is a perfect candidate for this approach because it lacks the symmetries required by older manual construction methods. The QECC toolkit uses a "Fault-Set Guided Synthesis" algorithm to find these circuits automatically by backtracking whenever a constraint is violated.

```{code-cell} ipython3
from mqt.qecc.codes import SquareOctagonColorCode

soc = SquareOctagonColorCode(5)
c1 = heuristic_prep_circuit(soc)
#BUG: Distance is not properly initialized
c1.code.distance = soc.distance
c1.code.x_distance = soc.distance
c1.code.z_distance = soc.distance
c1c2c3c4 = synthesize_four_ft_circuits(c1)

```

## Simulation and Verification

Once the four circuits are generated, they form a complete fault-tolerant gadget. The expected behavior is that the logical error rate should scale with $O(p^{\lceil d/2 \rceil})$, and the circuit should satisfy strict fault tolerance (no error of weight $t \le (d-1)/2$ leads to a logical failure).

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis.simulation import SteaneNDFTStatePrepSimulator
simulation = SteaneNDFTStatePrepSimulator(
    circ1=c1c2c3c4[0].circ.to_qiskit_circuit(),
    circ2=c1c2c3c4[1].circ.to_qiskit_circuit(),
    circ3=c1c2c3c4[2].circ.to_qiskit_circuit(),
    circ4=c1c2c3c4[3].circ.to_qiskit_circuit(),
    code=soc
)
ps = [0.007,0.006,0.005, 0.004, 0.003, 0.002, 0.001]
# simulation.plot_state_prep(ps, min_errors=10, p_idle_factor=0.01)

```

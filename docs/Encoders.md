---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 300
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Encoding Circuit Synthesis

## Encoder Circuit Synthesis for CSS Codes

QECC provides functionality for synthesizing encoding circuits of arbitrary
Stabilizer codes. A collection of pre-synthesized encoding circuits can be found
at [QECirc.com](https://qecirc.com/). An encoder for an $[[n,k,d]]$ code is an
isometry that encodes $k$ logical qubits into $n$ physical qubits.

Let's consider the synthesis of the encoding circuit of the $[[7,1,3]]$ Steane
code.

```{code-cell} ipython3
from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import (
    depth_optimal_encoding_circuit,
    gate_optimal_encoding_circuit
)

def print_code_operators(code, label="Code"):
    """Helper function to print stabilizers and logical operators of a code."""
    print(f"{label} Stabilizers:")
    for stab in code.stabs_as_pauli_strings():
        print(f"  {stab}")
    print(f"\n{label} X Logicals:")
    for x_log in code.x_logicals_as_pauli_strings():
        print(f"  {x_log}")
    print(f"{label} Z Logicals:")
    for z_log in code.z_logicals_as_pauli_strings():
        print(f"  {z_log}")

steane_code = CSSCode.from_code_name("steane")

print_code_operators(steane_code, "Steane Code")
```

There is not a unique encoding circuit but usually we would like to obtain an
encoding circuit that is optimal with respect to some metric. QECC has
functionality for synthesizing gate- or depth-optimal encoding circuits.

Under the hood, this uses the SMT solver [z3](https://github.com/Z3Prover/z3).
Of course this method scales only up to a few qubits. Synthesizing depth-optimal
circuits is usually faster than synthesizing gate-optimal circuits.

```{note}
These optimal encoders are thin convenience wrappers around the general
exact-synthesis engine. See {doc}`ExactSynthesis` for the full interface —
non-CSS codes, gate-count vs. depth objectives, custom gate sets, and
two-qubit-gate minimization.
```

```{code-cell} ipython3
depth_opt = depth_optimal_encoding_circuit(steane_code, max_timeout=2)
q_enc = depth_opt.inputs()

print(f"Encoding qubits are qubits {q_enc}.")
print(f"Circuit has depth {depth_opt.depth()}.")
print(f"Circuit has {depth_opt.num_cnots()} CNOTs.")

depth_opt.draw('mpl')
```

```{code-cell} ipython3
gate_opt = gate_optimal_encoding_circuit(steane_code, max_timeout=2)
q_enc = gate_opt.inputs()

print(f"Encoding qubits are qubits {q_enc}.")
print(f"Circuit has depth {gate_opt.depth()}.")
print(f"Circuit has {gate_opt.num_cnots()} CNOTs.")

gate_opt.draw('mpl')
```

QECC obtains optimal solutions for circuits by iteratively trying out different
parameters to close in on the optimum. Each run will only be run until the
number of seconds specified by `max_timeout`. If a solution is found in this
time it is returned. Otherwise, `None` will be returned.

In addition to the circuit, the synthesis methods also return the encoding
qubits. All other qubits are assumed to be initialized in the $|0\rangle$ or
$|+\rangle$ states.

For larger codes, synthesizing optimal circuits is not feasible. For this case,
QECC provides more scalable heuristic synthesis methods that can target the
optimization of two-qubit gates or depth.

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import (
    synthesize_encoding_circuit,
)

heuristic_circ = synthesize_encoding_circuit(steane_code)
q_enc = heuristic_circ.inputs()

print(f"Encoding (logical input) qubits: {q_enc}")
print(f"Circuit has depth {heuristic_circ.depth()}.")
print(f"Circuit has {heuristic_circ.num_cnots()} CNOTs.")

heuristic_circ.draw('mpl')
```

By default the heuristic synthesis tries to optimize for two-qubit gate count.
We can also tell the synthesis to optimize for depth.

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import (
    SynthesisConfig
)

config = SynthesisConfig(optimization_criterion="depth")
heuristic_circ = synthesize_encoding_circuit(steane_code, config=config)
q_enc = heuristic_circ.inputs()

print(f"Encoding (logical input) qubits: {q_enc}")
print(f"Circuit has depth {heuristic_circ.depth()}.")
print(f"Circuit has {heuristic_circ.num_cnots()} CNOTs.")

heuristic_circ.draw('mpl')
```

The `inputs()` method returns a list of physical qubit indices representing the
encoded logical information. All other qubits are ancillas initialized in the
$|0\rangle$ or $|+\rangle$ state.

### Extracting the Code from an Encoding Circuit

Given an encoding circuit, we can extract the stabilizer code it implements
using the `get_code()` method. This is useful for verifying that a synthesized
circuit correctly implements the desired code.

```{code-cell} ipython3
encoder = synthesize_encoding_circuit(steane_code)
circuit_code = encoder.get_code()

print(f"Original code: n={steane_code.n}, k={steane_code.k}")
print(f"Circuit code: n={circuit_code.n}, k={circuit_code.k}")
print(f"\nCodes are equivalent: {steane_code.is_equivalent(circuit_code)}")

print("\n" + "="*60)
print_code_operators(steane_code, "Original Steane Code")
print("\n" + "="*60)
print_code_operators(circuit_code, "Circuit-Extracted Code")
```

The `is_equivalent` method checks whether two codes have the same stabilizer
group and logical basis (up to stabilizer equivalence).

### Mapping Logical Qubits to Physical Inputs

For codes with multiple logical qubits ($k > 1$), it's important to understand
which physical input qubit corresponds to which logical qubit of the code. The
synthesized encoding circuit may permute the logical qubits, so the order of
physical inputs returned by `inputs()` may not directly correspond to the order
of logical operators in the code definition.

Let's demonstrate this with the $[[15, 7, 3]]$ quantum Hamming code, which
encodes 7 logical qubits.

```{code-cell} ipython3
from mqt.qecc.codes import construct_quantum_hamming_code
hamming_code = construct_quantum_hamming_code(4) # [[15,7,3]] quantum Hamming code

print(f"Code parameters: n={hamming_code.n}, k={hamming_code.k}, d={hamming_code.distance}")
print_code_operators(hamming_code, "Hamming Code")

encoder = synthesize_encoding_circuit(hamming_code)
physical_inputs = encoder.inputs()

print(f"\nPhysical input qubits: {physical_inputs}")
print(f"Number of physical inputs: {len(physical_inputs)}")
```

The `logical_to_input_mapping` method returns a list where the $i$-th element is
the physical input qubit corresponding to the $i$-th logical qubit of the code.

```{code-cell} ipython3
mapping = encoder.logical_to_input_mapping(hamming_code)

if mapping is not None:
    print("Logical to Physical Input Mapping:")
    for logical_idx, physical_qubit in enumerate(mapping):
        print(f"  Logical qubit {logical_idx} -> Physical input qubit {physical_qubit}")
else:
    print("The encoder does not implement the given code.")
```

This mapping tells us that to encode logical qubit $i$, we should prepare the
state on physical qubit `mapping[i]`.

### Fixing Input States for Logical Qubits

It is additionally possible to synthesize an encoding circuit of a $[[n, k, d]]$
code, with $f < k$ arbitrary inputs explicitly fixed, thus effectively returning
a circuit for a $[[n, k-f, d]]$ code. Each input can be fixed with either a
$|0\rangle$ or $|+\rangle$ state.

```{code-cell} ipython3
input_states = {1: "0", 3:"+"}

free_input_circ = synthesize_encoding_circuit(hamming_code)
fixed_input_circ = synthesize_encoding_circuit(
    hamming_code,
    fixed_logical_qubits=input_states
)

print("Original circuit:\n"
    f"  Logical input qubits  : {free_input_circ.num_inputs()}\n"
    f"  Physical output qubits: {free_input_circ.num_outputs()}")
print("Fixed circuit:\n"
    f"  Logical input qubits  : {fixed_input_circ.num_inputs()}\n"
    f"  Physical output qubits: {fixed_input_circ.num_outputs()}")
```

## Tweaking Parameters for Heuristic Synthesis

Let's consider a slightly larger example, the $[[23,1,7]]$
[Golay code](https://errorcorrectionzoo.org/c/qubit_golay).

```{code-cell} ipython3
code = CSSCode.from_code_name("golay")

encoder = synthesize_encoding_circuit(code)
print(f"Encoding (logical input) qubits: {encoder.inputs()}")
print(f"Circuit has depth {encoder.depth()}.")
print(f"Circuit has {encoder.num_cnots()} CNOTs.")

encoder.draw(output='mpl', fold=False, scale=0.5)
```

The way the greedy synthesis works in QECC is by trying to reduce the check
matrix (or stabilizer tableau) of the code in question using as few elementary
matrix operations (gates) as possible. The synthesis is greedily guided by some
metric computed on the check matrix - the number of remaining entries in the
check matrix, for example. Since the synthesis algorithm makes local choices,
the search might go down branches of the search-tree leading to sub-optimal
solution. QECC also provides a costlier search procedure which tries to look
ahead which candidates in the search will lead to good results by completing the
entire greedy synthesis for these candidates and making a choice based on the
resulting circuits (as opposed to using the check matrix as a proxy for
estimating).

This search is generally costlier, but can lead to significantly better results.
We can tell the synthesis to perform rollout-based synthesis by setting the
appropriate flags in the config. `rollout` is an int parameter that determines
for how many layers the search should perform the rollout. If it is set to `0`,
no rollout is performed (default). If it is set to `1`, the
`num_rollout_candidates` parameter determines for how many candidates per gate
in the search the rollout is performed. This determines how many complete
circuits are synthesized before the single gate is chosen leading to the locally
best circuit. If `enable_early_termination` is set to `True`, rollout is only
performed until no better solutions are found. In that case, the search returns
whatever the current best circuit is. If it is set to `False`, rollout will be
performed until the last gate of the search is placed. This will take longer but
leads to better results in general:

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import SynthesisConfig

config = SynthesisConfig(rollout=1, num_rollout_candidates=5, optimization_criterion="gates", enable_early_termination=False)

encoder = synthesize_encoding_circuit(code, config=config)
print(f"Encoding (logical input) qubits: {encoder.inputs()}")
print(f"Circuit has depth {encoder.depth()}.")
print(f"Circuit has {encoder.num_cnots()} CNOTs.")

encoder.draw(output='mpl', fold=False, scale=0.5)
```

## Encoder Circuit Synthesis for non-CSS Stabilizer Codes

QECC also supports encoding circuit synthesis for non-CSS stabilizer codes. For
these codes, encoding circuits are built from two-qubit transvections (see
arXiv:2503.14660 for details).

Let's consider the $[[5,1,3]]$ code.

```{code-cell} ipython3
from mqt.qecc import StabilizerCode
from mqt.qecc.circuit_synthesis import synthesize_encoding_circuit

stabs = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
x_logicals = ["XXXXX"]
z_logicals = ["ZZZZZ"]
five_qubit_code = StabilizerCode(stabs, x_logicals=x_logicals, z_logicals=z_logicals)

print("Stabilizers:")
for stab in stabs:
    print(f"  {stab}")
print("\nX Logical:")
print(f"  {x_logicals[0]}")
print("Z Logical:")
print(f"  {z_logicals[0]}")
```

The same `synthesize_encoding_circuit` function works for non-CSS codes. This
method returns a `CliffordIsometry` object.

```{code-cell} ipython3
encoder = synthesize_encoding_circuit(five_qubit_code)
encoding_qubits = encoder.inputs()

print(f"Encoding qubits (logical to physical mapping): {encoding_qubits}")
print(f"Circuit has two-qubit depth {encoder.depth()}.")
print(f"Circuit has {encoder.num_two_qubit_gates()} two-qubit gates.")

encoder.draw(output='mpl', fold=False)
```

For displaying the circuit, the transvections are decomposed into CZ and
single-qubit Clifford gates.

For non-CSS codes, depth-optimal synthesis is also available:

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import depth_optimal_encoding_circuit_non_css

# Search for a depth-optimal encoder with at most six two-qubit gates. `min_depth` skips the
# (expensive) proofs that no shallower circuit exists.
result = depth_optimal_encoding_circuit_non_css(five_qubit_code, min_depth=4, max_depth=8, max_two_qubit_gates=6)

if isinstance(result, str):
    raise RuntimeError(f"No non-CSS encoder found: {result}")
encoder = result
encoding_qubits = encoder.inputs()

print(f"Encoding qubits (logical to physical mapping): {encoding_qubits}")
print(f"Circuit has two-qubit depth {encoder.depth()}.")
print(f"Circuit has {encoder.num_two_qubit_gates()} two-qubit gates.")

encoder.draw(output='mpl', fold=False)
```

This method uses SMT-based synthesis to find a depth-optimal encoding circuit,
similar to the CSS case. The `max_depth` parameter limits the search depth. If
no solution is found, it returns `"UNSAT"`.

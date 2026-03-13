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

# Encoder Circuit Synthesis for CSS Codes

QECC provides functionality for synthesizing encoding circuits of arbitrary Stabilizer codes. An encoder for an $[[n,k,d]]$ code is an isometry that encodes $k$ logical qubits into $n$ physical qubits.

Let's consider the synthesis of the encoding circuit of the $[[7,1,3]]$ Steane code.

```{code-cell} ipython3
from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import (
    depth_optimal_encoding_circuit,
    gate_optimal_encoding_circuit
)

steane_code = CSSCode.from_code_name("steane")

print("Stabilizers:\n")
print(steane_code.stabs_as_pauli_strings())
print("\nLogicals:\n")
print(steane_code.x_logicals_as_pauli_strings())
print(steane_code.z_logicals_as_pauli_strings())
```

There is not a unique encoding circuit but usually we would like to obtain an encoding circuit that is optimal with respect to some metric. QECC has functionality for synthesizing gate- or depth-optimal encoding circuits.

Under the hood, this uses the SMT solver [z3](https://github.com/Z3Prover/z3). Of course this method scales only up to a few qubits. Synthesizing depth-optimal circuits is usually faster than synthesizing gate-optimal circuits.

```{code-cell} ipython3
depth_opt = depth_optimal_encoding_circuit(steane_code, max_timeout=2)
q_enc = depth_opt.get_uninitialized()

print(f"Encoding qubits are qubits {q_enc}.")
print(f"Circuit has depth {depth_opt.depth()}.")
print(f"Circuit has {depth_opt.num_cnots()} CNOTs.")

depth_opt.draw('mpl')
```

```{code-cell} ipython3
gate_opt = gate_optimal_encoding_circuit(steane_code, max_timeout=2)
q_enc = gate_opt.get_uninitialized()

print(f"Encoding qubits are qubits {q_enc}.")
print(f"Circuit has depth {gate_opt.depth()}.")
print(f"Circuit has {gate_opt.num_cnots()} CNOTs.")

gate_opt.draw('mpl')
```

QECC obtains optimal solutions for circuits by iteratively trying out different parameters to close in on the optimum. Each run will only be run until the number of seconds specified by `max_timeout`. If a solution is found in this time it is returned. Otherwise, `None` will be returned.

In addition to the circuit, the synthesis methods also return the encoding qubits. All other qubits are assumed to be initialized in the $|0\rangle$ state.

For larger codes, synthesizing optimal circuits is not feasible. In this case, QECC provides a heuristic synthesis method that tries to use as few CNOTs with the lowest depth as possible.

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import (
    synthesize_encoding_circuit,
)

heuristic_circ = synthesize_encoding_circuit(steane_code)
q_enc = heuristic_circ.inputs()

print(f"Messaging (logical input) qubits: {q_enc}")
print(f"Circuit has depth {heuristic_circ.depth()}.")
print(f"Circuit has {heuristic_circ.num_cnots()} CNOTs.")

heuristic_circ.draw('mpl')
```

By default the heuristic synthesis tries to optimize for two-qubit gate count. We can also tell the synthesis to optimize for depth.

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import (
    SynthesisConfig
)

config = SynthesisConfig(optimization_criterion="depth")
heuristic_circ = synthesize_encoding_circuit(steane_code)
q_enc = heuristic_circ.inputs()

print(f"Messaging (logical input) qubits: {q_enc}")
print(f"Circuit has depth {heuristic_circ.depth()}.")
print(f"Circuit has {heuristic_circ.num_cnots()} CNOTs.")

heuristic_circ.draw('mpl')
```

The `inputs()` method returns a list of physical qubit indices representing the encoded logical information. All other qubits are ancillas initialized in the $|0\rangle$ or $|+\rangle$ state.

# Encoder Circuit Synthesis for non-CSS Stabilizer Codes

QECC also supports encoding circuit synthesis for non-CSS stabilizer codes. For these codes, encoding circuits are built from two-qubit transvections (see arXiv:2503.14660 for details).

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

The same `synthesize_encoding_circuit` function works for non-CSS codes. This method returns a `CliffordIsometry` object.

```{code-cell} ipython3
encoder = synthesize_encoding_circuit(five_qubit_code)
message_qubits = encoder.inputs()

print(f"Message qubits (logical to physical mapping): {message_qubits}")
print(f"Circuit has two-qubit depth {encoder.depth()}.")
print(f"Circuit has {encoder.num_two_qubit_gates()} two-qubit gates.")

encoder.draw(output='mpl', fold=False)
```

For displaying the circuit, the transvections are decomposed into CZ and single-qubit Clifford gates.

For non-CSS codes, depth-optimal synthesis is also available:

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import depth_optimal_encoding_circuit_non_css

for d in range(3, 9):
    result = depth_optimal_encoding_circuit_non_css(five_qubit_code, max_depth=d, exact_two_qubit_count=True, max_two_qubit_gates=6)

    if result == "UNSAT":
        print(f"No solution for max_depth={d}")
    else:
        encoder = result
        break
print(f"Message qubits (logical to physical mapping): {message_qubits}")
print(f"Circuit has two-qubit depth {encoder.depth()}.")
print(f"Circuit has {encoder.num_two_qubit_gates()} two-qubit gates.")

encoder.draw(output='mpl', fold=False)
```

This method uses SMT-based synthesis to find a depth-optimal encoding circuit, similar to the CSS case. The `max_depth` parameter limits the search depth. If no solution is found, it returns `"UNSAT"`.

## Tweaking Parameters for Heuristic Synthesis

Let's consider a slightly larger example, the $[[23,1,7]]$ [Golay code](https://errorcorrectionzoo.org/c/qubit_golay).

```{code-cell} ipython3
code = CSSCode.from_code_name("golay")

encoder = synthesize_encoding_circuit(code)
print(f"Messaging (logical input) qubits: {encoder.inputs()}")
print(f"Circuit has depth {encoder.depth()}.")
print(f"Circuit has {encoder.num_cnots()} CNOTs.")

encoder.draw(output='mpl', fold=False, scale=0.5)
```

The way the greedy synthesis works in QECC is by trying to reduce the check matrix (or stabilizer tableau) of the code in question using as few elementary matrix operations (gates) as possible. The synthesis is greedily guided by some metric computed on the check matrix - the number of remaining entries in the check matrix, for example. Since the synthesis algorithm makes local choices, the search might go down branches of the search-tree leading to sub-optimal solution. QECC also provides a costlier search procedure which tries to look ahead which candidates in the search will lead to good results by completing the entire greedy synthesis for these candidates and making a choice based on the resulting circuits (as opposed to using the check matrix as a proxy for estimating).

This search is generally costlier, but can lead to significantly better results. We can tell the synthesis to perform rollout-based synthesis by setting the appropriate flags in the config. `rollout` is an int parameter that determines for how many layers the search should perform the rollout. If it is set to `0`, no rollout is performed (default). If it is set to `1`, the `num_rollout_candidates` parameter determines for how many candidates per gate in the search the rollout is performed. This determines how many complete circuits are synthesized before the single gate is chosen leading to the locally best circuit. If `enable_early_termination` rollout is only performed until no better solutions are found. In that case, the search returns whatever the current best circuit is. If it is set to `False`, rollout will be performed until the last gate of the search is placed. This will take longer but leads to better results in general:

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import SynthesisConfig

config = SynthesisConfig(rollout=1, num_rollout_candidates=5, optimization_criterion="gates", enable_early_termination=False)

encoder = synthesize_encoding_circuit(code, config=config)
print(f"Messaging (logical input) qubits: {encoder.inputs()}")
print(f"Circuit has depth {encoder.depth()}.")
print(f"Circuit has {encoder.num_cnots()} CNOTs.")

encoder.draw(output='mpl', fold=False, scale=0.5)
```

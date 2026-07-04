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

# Exact Circuit Synthesis

QECC provides an SMT/SAT-based exact synthesis engine for finding provably optimal Clifford circuits. It supports both CSS and non-CSS stabilizer codes and can optimize for gate count or circuit depth.

The entry point is `synthesize_isometry_exact` from `mqt.qecc.circuit_synthesis.exact`.

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis.exact import (
    synthesize_isometry_exact,
    Objective,
    SynthesisStatus,
    TargetKind,
)
```

## CSS Encoding Circuits

For CSS codes, encoding circuits consist only of CNOT gates (plus ancilla initialization). The target is specified as a `CheckMatrix` representing the stabilizer generators, and the logicals are provided as a `CheckMatrix` of logical operators.

Let us synthesize an encoder for the $[[4,2,2]]$ iceberg code — a small CSS code that encodes two logical qubits.

```{code-cell} ipython3
from mqt.qecc.codes import construct_iceberg_code
from mqt.qecc.codes.pauli import CheckMatrix

code = construct_iceberg_code(2)  # [[4,2,2]] iceberg code
print(f"[[{code.n},{code.k},{code.distance}]] iceberg code")
print(f"Hx =\n{code.Hx}")
print(f"Lx =\n{code.Lx}")
```

To synthesize a gate-count-optimal encoder, pass the X-check matrix as target and the logical X operators:

```{code-cell} ipython3
hx = CheckMatrix(code.Hx, pauli_type="X")
lx = CheckMatrix(code.Lx, pauli_type="X")

result = synthesize_isometry_exact(
    target=hx,
    target_kind=TargetKind.CSS_ISOMETRY,
    objective=Objective.GATE_COUNT,
    x_logicals=lx,
    lower_bound=0,
    upper_bound=6,
    timeout=60,
)

print(f"Status:         {result.status.value}")
print(f"Gate count:     {result.gate_count}")
print(f"Depth:          {result.depth}")
print(f"Proven optimal: {result.proven_optimal}")
print(f"Verified:       {result.verified}")
```

The synthesized circuit is a `CNOTCircuit` and can be drawn or serialized:

```{code-cell} ipython3
result.circuit.draw("mpl")
```

To optimize for depth instead, use `Objective.DEPTH`:

```{code-cell} ipython3
result_depth = synthesize_isometry_exact(
    target=hx,
    target_kind=TargetKind.CSS_ISOMETRY,
    objective=Objective.DEPTH,
    x_logicals=lx,
    lower_bound=0,
    upper_bound=6,
    timeout=60,
)

print(f"Status:         {result_depth.status.value}")
print(f"Gate count:     {result_depth.gate_count}")
print(f"Depth:          {result_depth.depth}")
print(f"Proven optimal: {result_depth.proven_optimal}")
```

## Non-CSS Encoding Circuits

For non-CSS stabilizer codes, encoding circuits are built from the full Clifford gate set. The target is specified as a `StabilizerTableau`, and the logicals are provided as `StabilizerTableau` objects.

The gate set for non-CSS synthesis defaults to $\{H, S, \text{CX}\}$ (standard Clifford). The extended gate set $\{H, S, \sqrt{X}, \text{CX}, \text{CZ}\}$ can find shorter circuits:

```{code-cell} ipython3
from mqt.qecc.codes.pauli import StabilizerTableau
from mqt.qecc.circuit_synthesis.exact import get_clifford_extended_gate_set

stabs = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
x_log = ["XXXXX"]
z_log = ["ZZZZZ"]

stab_tab = StabilizerTableau.from_pauli_strings(stabs)
x_log_tab = StabilizerTableau.from_pauli_strings(x_log)
z_log_tab = StabilizerTableau.from_pauli_strings(z_log)

result = synthesize_isometry_exact(
    target=stab_tab,
    target_kind=TargetKind.CLIFFORD_ISOMETRY,
    objective=Objective.DEPTH,
    x_logicals=x_log_tab,
    z_logicals=z_log_tab,
    gate_set=get_clifford_extended_gate_set(),
    lower_bound=3,
    upper_bound=7,
    use_symmetry_breaking=True,
    timeout=120,
)

print(f"Status:         {result.status.value}")
print(f"Two-qubit depth: {result.depth}")
print(f"Total gates:    {result.gate_count}")
print(f"Proven optimal: {result.proven_optimal}")
print(f"Verified:       {result.verified}")
```

The synthesized circuit is a `CliffordIsometry`:

```{code-cell} ipython3
result.circuit.draw("mpl")
```

## CSS State Synthesis

Exact synthesis also prepares CSS stabilizer states directly. For a state target only the check matrix is needed (no logicals).

The GHZ state $|GHZ\rangle \propto |000\rangle + |111\rangle$ is the cat ($|+\rangle_L$) state of the three-qubit repetition code: it is stabilized by the code's $Z$-checks $\{ZZI, IZZ\}$ together with $X_L = XXX$. We therefore pass those $Z$-checks as a CSS state target and find the shortest-depth preparation circuit:

```{code-cell} ipython3
import numpy as np

# Z-checks of the 3-qubit repetition code (ZZI and IZZ)
ghz_checks = CheckMatrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8), pauli_type="Z")

result = synthesize_isometry_exact(
    target=ghz_checks,
    target_kind=TargetKind.CSS_STATE,
    objective=Objective.DEPTH,
    lower_bound=0,
    upper_bound=5,
    timeout=30,
)

print(f"Status:         {result.status.value}")
print(f"Depth:          {result.depth}")
print(f"Gate count:     {result.gate_count}")
print(f"Proven optimal: {result.proven_optimal}")
result.circuit.draw("mpl")
```

## Gate Sets

The synthesis engine supports several Clifford gate sets for non-CSS targets. CSS synthesis always uses CNOT-only circuits and ignores the `gate_set` parameter.

| Factory function                   | Gates                                  | Notes                                |
| ---------------------------------- | -------------------------------------- | ------------------------------------ |
| `get_standard_clifford_gate_set()` | $H, S, \text{CX}$                      | Default for non-CSS synthesis        |
| `get_clifford_sx_gate_set()`       | $H, \sqrt{X}, \text{CX}$               | Replaces $S$ with $\sqrt{X} = HSH$   |
| `get_clifford_cz_gate_set()`       | $H, S, \text{CX}, \text{CZ}$           | Adds $\text{CZ}$ to the standard set |
| `get_clifford_extended_gate_set()` | $H, S, \sqrt{X}, \text{CX}, \text{CZ}$ | Full extended set                    |

A larger gate set gives the solver more freedom and can yield shorter circuits, at the cost of a larger search space per depth/count bound.

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis.exact import (
    get_standard_clifford_gate_set,
    get_clifford_cz_gate_set,
    get_clifford_extended_gate_set,
)

print("Standard gate set:", list(get_standard_clifford_gate_set().keys()))
print("CZ gate set:      ", list(get_clifford_cz_gate_set().keys()))
print("Extended gate set:", list(get_clifford_extended_gate_set().keys()))
```

## Target Kinds

The `TargetKind` enum selects the synthesis problem:

| Value               | Target type                     | Required arguments                                       |
| ------------------- | ------------------------------- | -------------------------------------------------------- |
| `CSS_ISOMETRY`      | CSS encoding isometry           | `CheckMatrix` target + `x_logicals` or `z_logicals`      |
| `CSS_STATE`         | CSS stabilizer state            | `CheckMatrix` target only                                |
| `CLIFFORD_ISOMETRY` | Full Clifford encoding isometry | `StabilizerTableau` target + `x_logicals` + `z_logicals` |
| `CLIFFORD_UNITARY`  | Full $n$-qubit Clifford unitary | `StabilizerTableau` target + `x_logicals` + `z_logicals` |
| `STABILIZER_STATE`  | Stabilizer state (any)          | `StabilizerTableau` target only                          |

## Search Configuration

### Bounds and timeouts

`synthesize_isometry_exact` performs an exhaustive search from `lower_bound` to `upper_bound` (inclusive). The first feasible bound returns a solution. If all bounds are infeasible the result has status `UNSAT`. A per-bound solver timeout can be set in seconds:

```python
result = synthesize_isometry_exact(
    ...,
    lower_bound=0,
    upper_bound=20,
    timeout=60,  # give each bound up to 60 seconds
)
```

If the solver times out at any bound, the search returns immediately with status `TIMEOUT`.

### Symmetry breaking

Symmetry-breaking constraints prune the SAT search space by forbidding obviously redundant gate sequences (adjacent identical self-inverse gates, unnecessary idle slots). Enable it with `use_symmetry_breaking=True`:

```python
result = synthesize_isometry_exact(
    ...,
    use_symmetry_breaking=True,
)
```

Symmetry breaking is most effective for larger problems where the raw SAT instance is expensive. It is safe to combine with any gate set.

### Exponential-backoff search

For large instances where a single per-bound timeout is too aggressive, the exponential-backoff strategy can find good solutions faster:

```python
result = synthesize_isometry_exact(
    ...,
    use_exponential_backoff=True,
    min_timeout=1,  # start with 1 second per bound
    timeout=3600,  # maximum per-bound budget
)
```

The strategy works in two phases:

1. **Ascending phase** — scan from `lower_bound` to `upper_bound` with `min_timeout` per bound. Bounds proven UNSAT are dropped permanently. Timed-out bounds are retried with a doubled budget after each pass, up to `timeout`.
2. **Descending phase** — once a SAT solution is found at bound $b$, descend from $b{-}1$ with the maximum budget to tighten the result.

`result.proven_optimal` is `True` only when all smaller bounds were confirmed UNSAT.

## Interpreting `SynthesisResult`

The `SynthesisResult` object returned by `synthesize_isometry_exact` carries all relevant metadata:

| Attribute        | Type                                      | Description                                             |
| ---------------- | ----------------------------------------- | ------------------------------------------------------- |
| `status`         | `SynthesisStatus`                         | `SUCCESS`, `UNSAT`, `TIMEOUT`, or `ERROR`               |
| `circuit`        | `CliffordIsometry \| CNOTCircuit \| None` | Synthesized circuit, or `None` if failed                |
| `gate_count`     | `int \| None`                             | Total non-identity non-Pauli gate count                 |
| `depth`          | `int \| None`                             | Two-qubit-gate depth                                    |
| `proven_optimal` | `bool`                                    | `True` when all smaller bounds were proven UNSAT        |
| `verified`       | `bool`                                    | `True` when the circuit was verified against the target |
| `message`        | `str`                                     | Human-readable status message                           |

```{code-cell} ipython3
result = synthesize_isometry_exact(
    target=hx,
    target_kind=TargetKind.CSS_ISOMETRY,
    objective=Objective.GATE_COUNT,
    x_logicals=lx,
    lower_bound=0,
    upper_bound=6,
    timeout=60,
)

print(f"status:         {result.status}")
print(f"gate_count:     {result.gate_count}")
print(f"depth:          {result.depth}")
print(f"proven_optimal: {result.proven_optimal}")
print(f"verified:       {result.verified}")
print(f"message:        {result.message}")
```

## Secondary Two-Qubit Gate Minimization

Depth-optimal synthesis may leave room to reduce the two-qubit gate count while keeping the depth fixed. The `max_two_qubit_gates` parameter bounds the number of two-qubit gates at a fixed depth, enabling a descent that finds the depth-optimal circuit with fewest two-qubit gates:

```{code-cell} ipython3
# Step 1: start from the depth-optimal circuit (depth 5 was established above)
depth_result = synthesize_isometry_exact(
    target=stab_tab,
    target_kind=TargetKind.CLIFFORD_ISOMETRY,
    objective=Objective.DEPTH,
    x_logicals=x_log_tab,
    z_logicals=z_log_tab,
    gate_set=get_clifford_extended_gate_set(),
    lower_bound=5,
    upper_bound=5,
    use_symmetry_breaking=True,
    timeout=30,
)

d_star = depth_result.depth
tq_count = depth_result.circuit.num_two_qubit_gates()
best_result = depth_result

print(f"Depth-optimal circuit: depth={d_star}, TQ gates={tq_count}")

# Step 2: descend on two-qubit gate count at fixed depth d_star
tq_proven_optimal = False
for max_tq in range(tq_count - 1, -1, -1):
    tq_result = synthesize_isometry_exact(
        target=stab_tab,
        target_kind=TargetKind.CLIFFORD_ISOMETRY,
        objective=Objective.DEPTH,
        x_logicals=x_log_tab,
        z_logicals=z_log_tab,
        gate_set=get_clifford_extended_gate_set(),
        lower_bound=d_star,
        upper_bound=d_star,
        max_two_qubit_gates=max_tq,
        timeout=6,
    )
    if tq_result.status == SynthesisStatus.SUCCESS:
        best_result = tq_result
        tq_count = max_tq
    elif tq_result.status == SynthesisStatus.UNSAT:
        tq_proven_optimal = True
        break
    else:
        break  # timeout: keep current best

print(f"\nMinimized circuit: depth={d_star}, TQ gates={tq_count}")
print(f"TQ count proven optimal: {tq_proven_optimal}")
```

## Storing and Reloading Circuits

Synthesized circuits can be serialized to Stim circuit strings for storage (e.g., in JSONL files or databases) and reloaded later:

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis.circuits import CliffordIsometry, CNOTCircuit
import stim

# Serialize
circuit_str = str(best_result.circuit.to_stim_circuit())
print("Serialized circuit:")
print(circuit_str)
```

```{code-cell} ipython3
# Reload a CliffordIsometry
stim_circ = stim.Circuit(circuit_str)
reloaded = CliffordIsometry.from_stim_circuit(stim_circ)

print(f"Reloaded: depth={reloaded.depth()}, TQ gates={reloaded.num_two_qubit_gates()}")
reloaded.draw("mpl")
```

CSS circuits (returned as `CNOTCircuit`) use the same interface:

```{code-cell} ipython3
cnot_str = str(result.circuit.to_stim_circuit())
reloaded_cnot = CNOTCircuit.from_stim_circuit(stim.Circuit(cnot_str))
print(f"Reloaded CSS circuit: depth={reloaded_cnot.depth()}, CNOTs={reloaded_cnot.num_cnots()}")
```

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

# Fault tolerant state preparation of Pauli eigenstates for CSS codes

The QECC package contains functionality for synthesizing and simulating fault
tolerant and non-fault tolerant state preparation circuits for Pauli eigenstates
of CSS codes. Currently, it supports synthesizing circuits for preparing any
logical Pauli eigenstate of the form $\{|s\rangle_L \mid s \in \{0,+\}^k\}$ of
arbitrary $[[n,k,d]]$ CSS codes. A collection of pre-synthesized state
preparation circuits can be found at [QECirc.com](https://qecirc.com/)

## Synthesizing non-FT state preparation circuits

A non-fault tolerant preparation circuit can be generated directly from a CSS
code. Let's consider the [Steane code](https://errorcorrectionzoo.org/c/steane)
which is a $[[7, 1, 3]]$ color code.

```{code-cell} ipython3
from mqt.qecc import CSSCode

steane_code = CSSCode.from_code_name("Steane")
print(steane_code.stabs_as_pauli_strings())
```

A state preparation circuit for the logical $|0\rangle_L$ of this code is a
circuit that generates a state that is stabilized by all of the above Pauli
operators and the logical $Z_L$ operator of the Steane code.

The code is small enough that we can generate a CNOT-optimal state preparation
circuit for it:

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import gate_optimal_prep_circuit

non_ft_sp = gate_optimal_prep_circuit(steane_code, zero_state=True, max_timeout=2)

non_ft_sp.circ.draw(output="mpl", initial_state=True)
```

We see that the minimal number of CNOTs required to prepare the logical
$|0\rangle_L$ circuit of the Steane code is $8$.

```{note}
`gate_optimal_prep_circuit` (and its depth-optimal counterpart) are thin
convenience wrappers around the general exact-synthesis engine. See
{doc}`ExactSynthesis` for the full interface, including custom gate sets and
depth optimization.
```

## Synthesizing FT state preparation circuits

The circuit above is not fault-tolerant. For example, an $X$ error on qubit
$q_1$ before the last CNOT propagates to a weight $2$ X error on $q_1$ and
$q_2$. This is to be expected since we apply two-qubit gates between the qubits
of a single logical qubit.

A common method to turn a non-FT protocol into a fault tolerant one is through
post-selection. We can try to detect whether an error was propagated through the
circuit and restart the preparation in case of a detection event. A circuit
performing such measurements is called a _verification circuit_.

Verification circuits need to be carefully constructed such that only
stabilizers of the code are measured and no more measurements are performed than
necessary. Finding good verification circuits is NP-complete.

QECC can automatically generate optimal verification circuits.

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import gate_optimal_verification_circuit

ft_sp = gate_optimal_verification_circuit(non_ft_sp)

ft_sp.draw(output="mpl", initial_state=True)
```

We have just automatically generated the known FT state preparation circuit for
the Steane code: [^1]. We see that if an X error happens on qubit $q_1$ before
the last CNOT causes the verification circuit to measure a $-1$.

## Simulating state preparation circuits

If we want to see the probability of a logical error happening after
post-selecting, QECC provides simulation utilities that can quickly generate
results. We can simulate the non-FT and FT circuits and compare the results.

[^1]: <https://www.nature.com/articles/srep19578>

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import VerificationNDFTStatePrepSimulator, CircuitLevelNoiseIdlingParallel

p = 0.05
noise = CircuitLevelNoiseIdlingParallel(p_tqg=p, p_sqg=p, p_init=p, p_meas=p, p_idle=p/100)

non_ft_simulator = VerificationNDFTStatePrepSimulator(
    non_ft_sp.circ, code=steane_code, zero_state=True
)
ft_simulator = VerificationNDFTStatePrepSimulator(
    ft_sp, code=steane_code, zero_state=True
)


pl_non_ft, ra_non_ft, _, _ = non_ft_simulator.logical_error_rate(noise, min_errors=10)
pl_ft, ra_ft, _, _ = ft_simulator.logical_error_rate(noise, min_errors=10)

print(f"Logical error rate for non-FT state preparation: {pl_non_ft}")
print(f"Logical error rate for FT state preparation: {pl_ft}")
```

The error rates seem quite close to each other. To properly judge the fault
tolerance of the circuits we want to look at how the logical error rate scales
with the physical error rate.

```{code-cell} ipython3
ps = [0.1, 0.05, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001]

non_ft_simulator.plot_state_prep(ps, min_errors=50, p_idle_factor=0.01)
ft_simulator.plot_state_prep(ps, min_errors=50, p_idle_factor=0.01)
```

Indeed we observe a quadratic scaling for the fault tolerant state preparation
circuit while the logical error rate scales linearly for the non-fault tolerant
state preparation.

## Beyond distance 3

Distance 3 circuits are particularly simple for fault tolerant state preparation
because for the $|0\rangle_L$ we can completely ignore Z errors. Due to error
degeneracy any Z error is equivalent to a weight 1 or 0 error.

Additionally one has to pay special attention to the order of measurements in
the verification circuits when more than one independent error in the state
preparation circuit is considered.

Because both error types are considered, the verification circuit now measures
both X- and Z-stabilizers. Unfortunately a Z error in an X measurement can
propagate to the data qubits and vice versa for Z measurements. Therefore, if we
check for Z errors after we have checked for X errors the measurements might
introduce more X errors on the data qubits. We can check those again but that
would just turn the situation around; now Z errors can propagate to the data
qubits.

Detecting such _hook errors_ can be achieved via flag-fault tolerant stabilizer
measurements [^2]. Usually, information from such hook errors is used to augment
an error correction scheme. But we can also use these flags as additional
measurements on which we post-select. If one of the flags triggers, this
indicates that a hook error happened and we reset.

By default QECC automatically performs such additional measurements when
necessary. The general construction is sketched in the following figure.

```{image} images/full_ft_scheme.svg
:width: 100%
:alt: Construction of non-deterministic fault-tolerant state preparation circuits
```

Let's consider a larger code to illustrate the point. The
[square-octagon color code](https://errorcorrectionzoo.org/c/488_color) is
defined on the following lattice:

```{image} images/488_color_code.svg
:width: 50%
:alt: Square-Octagon color code
```

The distance 5 code uses 17 qubits from this lattice, i.e., we have a
$[[17, 1, 5]]$ CSS code. Given the size of the code, synthesizing an optimal
state preparation circuit might take a long time. QECC also has a fast heuristic
state preparation circuit synthesis.

[^2]: <https://arxiv.org/abs/1708.02246>

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import heuristic_prep_circuit
from mqt.qecc.codes import SquareOctagonColorCode

cc = SquareOctagonColorCode(5)
cc_non_ft_sp = heuristic_prep_circuit(cc, state="all_zero", optimize_depth=True)

cc_non_ft_sp.circ.draw(output="mpl", initial_state=True, scale=0.7)
```

Even though optimal state preparation circuit synthesis seems out of range we
can still synthesize good verification circuits in a short time if we give an
initial guess on how many measurements we will need.

```{code-cell} ipython3
cc_ft_sp = gate_optimal_verification_circuit(
    cc_non_ft_sp, max_timeout=4, max_ancillas=3
)

cc_ft_sp.draw(output="mpl", initial_state=True, fold=-1, scale=0.2)
```

We see that the overhead for the verification overshadows the state preparation
by a large margin. But this verification circuit is still much smaller than the
naive variant of post-selecting on the stabilizer generators of the code.

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import naive_verification_circuit

cc_ft_naive = naive_verification_circuit(cc_non_ft_sp)

print(
    f"CNOTs required for naive FT state preparation: {cc_ft_naive.num_nonlocal_gates()}"
)
print(
    f"CNOTs required for optimized FT state preparation: {cc_ft_sp.num_nonlocal_gates()}"
)
```

We expect that the distance 5 color code should be prepared with a better
logical error rate than the Steane code. And this is indeed the case:

```{code-cell} ipython3
cc_simulator = VerificationNDFTStatePrepSimulator(cc_ft_sp, code=cc, zero_state=True)

ps = [0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008]

ft_simulator.plot_state_prep(
    ps, min_errors=50, p_idle_factor=0.01
)  # simulate Steane code as comparison
cc_simulator.plot_state_prep(ps, min_errors=50, p_idle_factor=0.01)
```

## Trading size against depth

The verification circuits above use as few CNOTs as possible, but they can be
deep. Shallower circuits need more measurements and ancilla qubits. The front
search finds this trade-off: for each CNOT depth, the fault tolerant circuit
with the fewest CNOTs plus measurements that it can find, if it has fewer than
every shallower one. We call this set of circuits a _front_.

### Check types

The circuits combine three kinds of checks, all used for post-selection:

- **Single checks:** one ancilla qubit measures one stabilizer, as in the
  verification circuits above. Cheap, but each check adds depth.
- **Flags:** an extra ancilla qubit on a single check catches a hook error, as
  in the previous section.
- **Block checks:** several stabilizers are measured at once through a prepared
  ancilla state on several qubits. This takes few CNOT layers but more ancilla
  qubits. The ancilla state can itself spread errors to the data, so it gets its
  own checks, again of any of the three kinds.

### How the search works

- A candidate circuit is a tree. The root is a non-fault tolerant preparation
  circuit; each node adds checks, flags or blocks, and each block's ancilla
  state is a child node.
- A tree search on several worker processes combines the check types and keeps
  every circuit that improves the front.
- Fast procedures, such as an integer program that chooses the checks, complete
  partial candidates. A final pass removes redundant checks, shares flags
  between checks and reorders gates.
- Every returned circuit is checked exactly: each set of up to $t$ faults (by
  default $\lfloor (d-1)/2 \rfloor$, at most 3) is either detected or leaves
  only correctable errors.

### Example

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import synthesize_ft_prep

front = synthesize_ft_prep(steane_code, seconds=20, workers=2)
for f in front:
    print(f"depth {f.depth}: {f.cnots} CNOTs, {f.measurements} measurements, {f.qubits} qubits")
```

Each entry holds a `stim.Circuit`. Its checks are `DETECTOR` instructions (a
check can be the parity of several measurements), and the simulator above
post-selects on them:

```{code-cell} ipython3
front_simulator = VerificationNDFTStatePrepSimulator(front[-1].circuit, code=steane_code, zero_state=True)
pl_front, ra_front, _, _ = front_simulator.logical_error_rate(noise, min_errors=10)
print(f"Logical error rate of the front circuit with the fewest operations: {pl_front}")
```

The circuits that long runs found for codes up to distance 7 are listed under
[Circuits and Evaluations](#circuits-and-evaluations). For the $[[17, 1, 5]]$
color code, the front compares with the verification circuit from above as
follows:

```{code-cell} ipython3
:tags: [remove-input]
import csv
from pathlib import Path

import matplotlib.pyplot as plt

index = Path("../scripts/ft_stateprep/front_search/index.csv")
rows = [r for r in csv.DictReader(index.open()) if r["code"] == "color_17_1_5"]
rows.sort(key=lambda r: (int(r["depth"]), int(r["ops"])))
rows = [r for i, r in enumerate(rows) if all(int(r["ops"]) < int(q["ops"]) for q in rows[:i])]  # front only
depths = [int(r["depth"]) for r in rows]
ops = [int(r["ops"]) for r in rows]

verified_ops = cc_ft_sp.count_ops().get("cx", 0) + cc_ft_sp.count_ops().get("measure", 0)
verified_depth = cc_ft_sp.depth(lambda instr: instr.operation.num_qubits == 2)

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.step(depths, ops, where="post", marker="o", label="front search")
ax.plot([verified_depth], [verified_ops], "s", label="verification circuit from above")
ax.set_xlabel("CNOT depth")
ax.set_ylabel("CNOTs + measurements")
ax.legend()
plt.show()
```

Notes:

- Codes of distance 5 and 7 need hours on many cores.
- `synthesize_ft_prep` can also start from given preparation circuits, prepare
  $|+\rangle_L$ and use other synthesis tools; see its API documentation.
- It starts worker processes, so in a script call it inside an
  `if __name__ == "__main__":` block.

## Deterministic state preparation ($d < 5$)

A possible disadvantage of the above approach is that the verification circuits
are non-deterministic. This means that we potentially have to run the circuit
multiple times to successfully prepare the state. This can be circumvented by
using the information gained from the verification measurements to possibly
identify the dangerous error and correct it. If this this is done for every
possible (single) error, we refer to the state preparation as deterministic
[^3].

For small codes ($d < 5$ i.e., we need to consider only a single error) this
problem is still tractable and can be solved in an optimal way using
satisfiability solvers. QECC can automatically generate deterministic state
preparation circuits for such codes.

[^3]: <https://arxiv.org/abs/2301.10017>

For this we come back to our $d=3$ Steane code.

```{code-cell} ipython3
non_ft_sp.circ.draw(output="mpl", initial_state=True)
```

And initialize an instance of the deterministic verification helper class to
facilitate the generation of the deterministic FT state preparation circuit.

```{code-cell} ipython3
from mqt.qecc.circuit_synthesis import DeterministicVerificationHelper

det_helper = DeterministicVerificationHelper(non_ft_sp)
```

Calling the `get_solution` method will generate the non-deterministic
verification circuit using either the optimal or the heuristic method discussed
above. The deterministic verification circuit, separated into X and Z correction
layers, is then generated using the satisfiability solver.

```{code-cell} ipython3
det_verify = det_helper.get_solution(use_optimal_verification=True)
det_verify_x, det_verify_z = det_verify
```

Such a `DeterministicVerification` object contains the stabilizer measurements
for the non-deterministic verification circuit, the stabilizer measurements for
the different deterministic verification circuits, depending on the
non-deterministic measurement outcomes, and the final correction Pauli operators
that need to be applied to the data qubits.

The non-deterministic verification measurements are stored in the `stabs`
attribute as a list of numpy arrays where each array represents a stabilizer
measurement.

```{code-cell} ipython3
det_verify_x.stabs
```

The deterministic verification measurements are stored in the `det_stabs`
attribute as a dictionary where the keys are the non-deterministic measurement
outcomes (converted to int) and the values is a tuple with the first element
being the deterministic stabilizer measurements and the second element being
again a dictionary with the Pauli corrections for the deterministic measurement
outcomes.

For example for the Steane code whenever the non-deterministic verification
triggers (1) the logical operator on qubits 2,3,6 hast to measured. If the
outcome is 1, a Pauli correction on qubit 3 has to be applied, otherwise no
correction is necessary.

```{code-cell} ipython3
det_verify_x.det_correction
```

For the case where the non-deterministic verification measurements need to be
flagged (not the case for the Steane code), the `hook_corrections` attribute
contains the additional stabilizer measurements and corrections in the same
format as the `det_stabs` attribute.

```{code-cell} ipython3
det_verify_x.hook_corrections
```

### Simulating deterministic state preparation circuits with Qsample

```{warning}
Due to [`qsample`](https://github.com/dpwinter/qsample) not being actively
maintained, this feature is optional. To use this feature, install QECC with the
`qsample` extra (`uv pip install "mqt.qecc[qsample]"`) or install qsample
locally. See the {doc}`installation instructions <installation>` for details.
```

The resulting `DeterministicVerification` object can be used to directly
simulate the deterministic state preparation circuit using the
[Qsample](https://github.com/dpwinter/qsample) under the hood. The
`NoisyDFTStatePrepSimulator` class automatically constructs a valid Qsample
protocol containing the correct circuits and conditional paths to simulate the
deterministic state preparation. The passed Error Model and simulation
parameters are directly passed to Qsample and explained in the
[Qsample documentation](https://dpwinter.github.io/qsample/). Similarly also the
Qsample callbacks can be used to e.g. directly plot the logical error rates.

```{code-block} python
:linenos:

from qsample import callbacks, noise

from mqt.qecc.circuit_synthesis import NoisyDFTStatePrepSimulator

error_model = noise.E1_1  # depolarizing error model
err_params = {"q": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]}
shots_dss = 2000
p_max = {"q": 0.01}
L = 1

qsample_sim = NoisyDFTStatePrepSimulator(
    non_ft_sp.circ, det_verify, steane_code, error_model
)
sampling_stats = qsample_sim.dss_logical_error_rates(
    err_params, p_max, L, shots_dss, callbacks=[callbacks.PlotStats()]
)
```

## Circuits and Evaluations

The circuits and benchmark scripts used for our non-deterministic work
{cite:labelpar}`peham2024automated`, can be found
[on GitHub](https://github.com/munich-quantum-toolkit/qecc/tree/main/scripts/ft_stateprep/eval)
and for the deterministic work {cite:labelpar}`schmid2024deterministic`
[on GitHub](https://github.com/munich-quantum-toolkit/qecc/tree/main/scripts/ft_stateprep/eval_det)
as well. The circuits found by long runs of the front search, for codes up to
distance 7, are
[on GitHub](https://github.com/munich-quantum-toolkit/qecc/tree/main/scripts/ft_stateprep/front_search),
together with the codes and a script that simulates them.

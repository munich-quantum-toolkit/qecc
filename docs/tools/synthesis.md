# Circuit Synthesis

Tools for synthesizing quantum circuits, including circuits that prepare
logical states fault-tolerantly.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Exact Circuit Synthesis
:link: ../ExactSynthesis
:link-type: doc

An SMT/SAT-based engine for provably optimal Clifford circuits. Supports CSS
and non-CSS stabilizer codes and optimizes for either gate count or depth.
:::

:::{grid-item-card} Encoding Circuit Synthesis
:link: ../Encoders
:link-type: doc

Encoding circuits for arbitrary stabilizer codes — the isometry mapping $k$
logical qubits into $n$ physical qubits.
:::

:::{grid-item-card} Pauli Eigenstates for CSS Codes
:link: ../StatePrep
:link-type: doc

Synthesis and simulation of fault-tolerant and non-fault-tolerant circuits
preparing logical Pauli eigenstates of arbitrary $[[n,k,d]]$ CSS codes.
:::

:::{grid-item-card} Cat State Preparation
:link: ../CatStates
:link-type: doc

Preparation of $|0\rangle^{\otimes w}+|1\rangle^{\otimes w}$ states, used for
Shor-style syndrome extraction, with verification against faults.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

../ExactSynthesis
../Encoders
../StatePrep
../CatStates
```

# Encoding-aware compilation for tetrahedral color codes

<!--
TODO before un-drafting:
- Align vocabulary and the definition of the encodings with the paper.
- Add the paper citation (refs.bib entry + {cite:labelpar}) once it is public.
- Add links to the implementation, the experiments repository and the Zenodo
  DOIs once they are public.
-->

```{note}
This page is a placeholder. Links to the paper, the implementation and the
experiment data will be added once the corresponding publication is available.
```

Logical qubits can be encoded in two tetrahedral color codes, denoted here R and
R'. Which logical operations are cheap depends on the encoding ("flavor") of the
qubits involved:

- $T$, $T^\dagger$, $S$ and $S^\dagger$ are cheap on R but require magic-state
  distillation on R'.
- A Hadamard switches a qubit between R and R'.
- A CNOT is cheap unless its control is in R' and its target in R. A CZ is cheap
  unless both qubits are in R'. The expensive cases require a round-robin
  implementation.

_Encoding-aware compilation_ tracks these flavors during ZX-calculus circuit
extraction and steers the extraction away from expensive operations.

## Where the code lives

Unlike the other functionality documented here, this work is not part of the
{code}`mqt.qecc` package, because it builds directly on the circuit extraction
of [PyZX](https://github.com/zxcalc/pyzx). The implementation and the
experiments will be linked here.

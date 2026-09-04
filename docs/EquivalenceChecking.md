---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Equivalence Checking

The _equivalence checking problem for quantum error-correction codes_ can be
defined as follows:

> Given two quantum error-correction codes $\text{C}$ and $\text{C}'$, do they
> define the same code space under a set of admissible transformations?

Equivalence notions define the admissible transformations. In the following, we
briefly explain the considered equivalence notions, and their restrictions in
our implementation.

```{code-cell} ipython3
from mqt.qecc import (
    StabilizerCode,
    are_local_clifford_equivalent,
    are_permutation_equivalent,
    is_local_clifford_equivalent_to_css,
)

base_code = StabilizerCode(generators=["XXXX","ZZII","IIZZ"])
```

## Equivalence Notions

### Exact Equivalence

Two stabilizer codes $\text{C}$ and $\text{C}'$ are considered exactly
equivalent, if and only if they have exactly the same code space, which is
precisely the case when they have the same stabilizer group.

This notion is checked with the function `equal_stabilizer_group`:

```{code-cell} ipython3
other_code = StabilizerCode(generators=["ZZZZ", "YYYY", "ZZII"])

print(base_code.equal_stabilizer_group(other_code))
```

```{note}
Unlike exact equivalence, the following permutation and local Clifford
equivalence use the unsigned stabilizer group: generator phases are ignored.
Thus, stabilizers only need to be mapped onto each other up to signs. Any
remaining signs can be corrected by a Pauli operation on the physical qubits.
```

### Permutation Equivalence

Two codes $\text{C}$ and $\text{C}'$ are considered permutation-equivalent if
and only if their code spaces are related by a permutation of the physical
qubits. Equivalently, their stabilizer groups differ by a common permutation of
the Pauli operators in their elements.

This notion is checked, and a potential witness permutation extracted, using the
function `are_permutation_equivalent`:

```{code-cell} ipython3
other_code = StabilizerCode(generators=["XXXX","ZIZI","IZIZ"])

print(are_permutation_equivalent(base_code, other_code))
```

### Local Clifford Equivalence

Two codes $\text{C}$ and $\text{C}'$ are regarded Local-Clifford-equivalent if
and only if they have the same code space up to a local Clifford operation on
the physical qubits. This is precisely the case when their stabilizer groups
differ by the conjugation with a common local Clifford operation on their
elements.

This notion is checked, and a potential witness operation extracted, using the
function `are_local_clifford_equivalent`:

```{code-cell} ipython3
other_code = StabilizerCode(generators=["ZXXX","XZII","IIZZ"])

print(are_local_clifford_equivalent(base_code, other_code))
```

#### Local Clifford Equivalence to a CSS Code

The module also supports deciding, whether a given stabilizer code is
local-Clifford-equivalent to an arbitrary CSS Code. As an example, this can be
used to enable a more optimized encoder circuit synthesis ({doc}`Encoders`) for
the equivalent CSS code and modifying the resulting circuit with the witness
operation to obtain an encoder circuit for the given stabilizer code.

For now, the current implementation does not support the extraction of a witness
operation.

```{code-cell} ipython3
stabilizer_code = StabilizerCode(generators=["YX"])

print(is_local_clifford_equivalent_to_css(stabilizer_code))
```

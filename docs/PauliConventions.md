# Pauli Conventions

This page pins down how Pauli operators, their phases, and stabilizer groups are
represented in `mqt.qecc.codes.core.pauli`. The conventions matter: a phase that
is read as if it were a sign bit produces silently wrong results rather than an
error.

## The representation

A `Pauli` on $n$ qubits is stored as a binary symplectic support $(x \mid z)$
together with a **phase exponent** $p \in \{0,1,2,3\}$:

$$
P = i^{p}\, X^{x} Z^{z}
\qquad\text{where}\qquad
X^{x} Z^{z} = \bigotimes_{j=1}^{n} X^{x_j} Z^{z_j}
$$

The exponent is stored in `Pauli.phase_exponent`; for a `PauliTableau` the
per-row exponents are in `PauliTableau.phase_exponents`.

The single-qubit letters follow from this with $Y = iXZ$:

| letter | $(x_j \mid z_j)$ | contribution to $p$ |
| ------ | ---------------- | ------------------- |
| `I`    | $(0 \mid 0)$     | 0                   |
| `X`    | $(1 \mid 0)$     | 0                   |
| `Z`    | $(0 \mid 1)$     | 0                   |
| `Y`    | $(1 \mid 1)$     | 1                   |

So a Hermitian Pauli with a `+` sign has $p = x \cdot z$, which is the number of
`Y` letters modulo four — **not** zero. `Pauli(support)` with no explicit
exponent picks exactly this canonical positive Hermitian choice.

## Exponents are not sign bits

`phase_exponent` carries four values, not two. The two are related by

$$
P = (-1)^{r}\, i^{\,x \cdot z}\, X^{x} Z^{z},
\qquad
r = \frac{p - x \cdot z}{2} \bmod 2
$$

Use the explicit converters rather than reaching for the raw exponent:

- `Pauli.sign()` / `PauliTableau.signs()` return the binary sign $r$, and raise
  `InvalidPauliError` on a non-Hermitian operator, which has no real sign.
- `Pauli.from_symplectic_and_sign(support, sign)` and
  `PauliTableau.phase_from_signs(matrix, signs)` go the other way.
- `Pauli.is_hermitian()` / `PauliTableau.is_hermitian()` test whether a sign
  exists at all.

## Multiplication needs a correction term

Because $Z^{z_1} X^{x_2} = (-1)^{z_1 \cdot x_2} X^{x_2} Z^{z_1}$, the product of
two Paulis is

$$
P_1 P_2 = i^{\,p_1 + p_2 + 2 (z_1 \cdot x_2)}
X^{x_1 \oplus x_2} Z^{z_1 \oplus z_2}
$$

The extra $2(z_1 \cdot x_2)$ is why
**XOR-ing symplectic rows and XOR-ing their signs is not Pauli multiplication**.
Concretely:

$$(X \otimes X)(Z \otimes Z) = -\,Y \otimes Y$$

while XOR-ing the two `+` signs would predict $+\,Y \otimes Y$. Note the
correction depends on the number of qubits: $(XXXX)(ZZZZ) = +YYYY$.

Consequences for anyone combining rows of a signed tableau:

- Use `PauliTableau.multiply_rows(target, source)`, never a raw XOR on
  `tableau.tableau.data` followed by an XOR on the phases.
- Use `pauli_row_echelon`, not `mod2.row_echelon`, whenever phases must survive
  the reduction. A plain mod-2 reduction is only safe on a CSS tableau, where
  the pivoting never combines an X-type row with a Z-type row and the correction
  term vanishes.
- `PauliTableau.independent_rows()` selects rows by support only and is
  explicitly phase-insensitive; do not use it to decide anything about signs.

## Which layer enforces what

The two layers deliberately allow different things:

- **`Pauli` / `PauliTableau` represent the full $n$-qubit Pauli group
  $\mathfrak{P}_n$.** Non-Hermitian elements such as `+iX` are legal and
  necessary: row reduction genuinely produces them as intermediates, since
  $X \cdot Z = -iY$.
- **`StabilizerCode` enforces the stabilizer conditions.** Its constructor
  rejects generators that do not commute, are not Hermitian, or together
  generate $-I$. Those checks — not the Pauli layer — are what guarantee a valid
  code.

Generators need **not** be independent. A redundant generating set is accepted
and kept as given, so `CSSCode` preserves the check matrices you pass in,
including redundant rows that matter for single-shot decoding and meta-checks.
Group-level comparisons (`equal_stabilizer_group`, `stabilizer_equivalent`,
`is_stabilizer`) compare the generated groups and are unaffected by redundancy.

## Subgroups and rank

`pauli_row_echelon` returns a `PauliRowEchelon`, whose `rank` is $\log_2 |G|$
for the generated subgroup $G$. This includes the central scalars, so it can
exceed the number of pivot columns:

| generators    | generated subgroup                                | order | `rank` |
| ------------- | ------------------------------------------------- | ----- | ------ |
| `["XX","ZZ"]` | $\{I, XX, ZZ, -YY\}$ — no scalars beyond $I$      | 4     | 2      |
| `["+iX"]`     | $\{I, iX, -I, -iX\}$ — one pivot, four scalars    | 4     | 2      |
| `["X","Z"]`   | $\{\pm I, \pm X, \pm Z, \pm iY\}$ — anticommuting | 8     | 3      |

The middle row has a single pivot column yet rank 2: the generator squares to
$-I$, so the subgroup contains scalars the support alone cannot account for. The
last row picks up $-I$ from the anticommutator.

To test many Paulis against one subgroup, compute the echelon once and call
`pauli_in_reduced_subgroup`; `PauliTableau.is_in_subgroup` redoes the
elimination on every call.

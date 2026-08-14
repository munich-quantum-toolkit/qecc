# Architecture

MQT QECC combines a shared mathematical representation of quantum
error-correcting codes with tools for circuit synthesis, state preparation,
decoding, and compilation. This page describes the main representation layers
and shows which parts of the toolkit currently build on them.

## Typical Data Flow

A typical workflow starts with a code construction and passes that code to a
specialized tool:

```text
Construction → Code → Tool → Result
```

The result depends on the tool. For example, synthesis and state-preparation
tools produce circuits, decoders produce corrections, and compilation tools
produce transformed circuits or schedules.

## Representation Layers

The shared QEC code model is assembled from layered mathematical building
blocks:

```text
Binary algebra
    → Symplectic vectors and vector spaces
    → Pauli-group elements
    → Stabilizer codes
    → CSS codes and specialized code families
```

| Layer | Main abstractions                                            | Responsibility                                                                                                          |
| ----: | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
|     1 | NumPy arrays and `mqt.qecc.mod2`                             | Binary matrices and linear algebra over $\mathbb{F}_2$.                                                                 |
|     2 | `SymplecticVector`, `SymplecticMatrix`, `symplectic_product` | The binary symplectic vector space used to encode Pauli support and determine commutation relations.                    |
|     3 | `Pauli`, `PauliTableau`, `CheckMatrix`                       | Signed Pauli operators, ordered collections of Pauli operators, and CSS-specific check matrices.                        |
|     4 | `StabilizerCode`                                             | Stabilizer generators and logical operators, together with syndrome, logical-operator, and code-equivalence operations. |
|     5 | `CSSCode`                                                    | A specialization of `StabilizerCode` with separate $X$ and $Z$ checks and logical operators.                            |
|     6 | `ColorCode`, `RotatedSurfaceCode`, and other constructions   | Concrete code families or functions that construct instances of the shared code model from a small set of parameters.   |

The layers describe a *builds on* relationship, not exclusively class
inheritance.

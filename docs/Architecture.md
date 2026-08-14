# Architecture

## Typical Data Flow

Construction -> Code -> Tool -> Result

## Representation

Bottom-up components building our representation of a Quantum Error Correction
Code.

### Binary Vectors

`numpy` arrays

### Symplectic Vectors

`SymplecticVector`, `SymplecticMatrix`, additionally `symplectic_product`

### Pauli Group Elements

`Pauli`, `PauliTableau`, `CheckMatrix`

### Codes

`StabilizerCode`

## Code Models

Building upon the shared `StabilizerCode`, what other more specialized code
families do we explicitly represent? `CSSCode`, `ColorCode`,
`HexagonalColorCode`, `SquareOctagonColorCode`

... and other constructions

## Component Integration?

potentially table with tools and supported representations etc

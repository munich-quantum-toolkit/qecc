# Hard-coded code instances

This directory holds concrete quantum error-correcting code instances that ship
with the package because they cannot (easily) be generated on the fly or are
convenient to have immediately available. It is the `instances/` layer of
`mqt.qecc.codes` (see also `core/` and `constructions/`).

## Named CSS codes

Each of the following sub-directories stores a code as two NumPy arrays,
`hx.npy` and `hz.npy` (the X- and Z-check matrices):

| Directory      | Code                       |
| -------------- | -------------------------- |
| `steane/`      | `[[7, 1, 3]]` Steane       |
| `shor/`        | `[[9, 1, 3]]` Shor         |
| `tetrahedral/` | `[[15, 1, 3]]` tetrahedral |
| `carbon/`      | `[[12, 2, 4]]` carbon      |
| `golay/`       | `[[23, 1, 7]]` Golay       |

These are the codes returned by `CSSCode.from_code_name(...)`, e.g.
`CSSCode.from_code_name("Steane")`.

## `lifted_product/`

Pre-generated check matrices for lifted-product codes, stored as sparse
`.npz` arrays. File names follow `lp_l=<lift>_h{x,z}.npz`.

## `misc/`

Example codes stored in the human-readable text formats accepted by
`CSSCode.from_file(...)`. They double as fixtures for the supported formats:

| File         | Format                                               |
| ------------ | ---------------------------------------------------- |
| `8_3_3.txt`  | Pauli-string stabilizers (one per line)              |
| `15_3_5.txt` | Python list notation (`[[1,0,...], ...]`)            |
| `30_6_5.txt` | NumPy-array notation (space-separated `[[1 0 ...]]`) |

File names follow the `n_k_d` convention (physical qubits, logical qubits,
distance).

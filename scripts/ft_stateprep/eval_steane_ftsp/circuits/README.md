# Circuits

State-preparation circuits of Weilandt, Peham, Wille, Phys. Rev. A (2026), doi:10.1103/y9ln-b4t9.

`<code>/<code>_heuristic_{0,1,2,3}.qasm` are the circuits $C_1, \dots, C_4$ of the Steane-type protocol.
They are identical to `reference/circuits/ours/` of the Zenodo record (doi:10.5281/zenodo.23183006).
`recipes.py` rebuilds them, and `check_reproduction.py` checks that it does.

| Paper      | Folder               |
| ---------- | -------------------- |
| [[17,1,5]] | `cc_4_8_8_d5`        |
| [[19,1,5]] | `cc_6_6_6_d5`        |
| [[25,1,5]] | `rotated_surface_d5` |
| [[20,2,6]] | `eve_20_2_6`         |
| [[31,1,7]] | `cc_4_8_8_d7`        |
| [[37,1,7]] | `cc_6_6_6_d7`        |

`flag_at_origin/` holds the baseline circuits used for comparison.

# Fault-tolerant state preparation circuits from the front search

This directory holds fault-tolerant circuits that prepare the logical all-zero
state of CSS codes. The front search of MQT QECC
(`mqt.qecc.circuit_synthesis.ft_search`) found them. For each code, the circuits
form a front: for each depth, the circuit with the fewest operations that the
search found, kept only if it has fewer operations than every shallower circuit.
Five further circuits are kept as well, although other circuits in this
directory beat them: `color_17_1_5` 77/7, 58/10 and 55/12, `bga_24_4_5` 123/20
and `color_31_1_7` 226/19 (operations/depth). A circuit for a code of distance
`d` tolerates `t = (d - 1) // 2` faults.

| Code                                  | Directory        |
| ------------------------------------- | ---------------- |
| [[9,1,3]] rotated surface code        | `surface_9_1_3`  |
| [[17,1,5]] 4.8.8 color code           | `color_17_1_5`   |
| [[25,1,5]] rotated surface code       | `surface_25_1_5` |
| [[24,4,5]] code                       | `bga_24_4_5`     |
| [[20,2,6]] code                       | `sd_20_2_6`      |
| [[23,1,7]] Golay code                 | `golay_23_1_7`   |
| [[31,1,7]] 4.8.8 color code           | `color_31_1_7`   |
| [[49,1,7]] rotated surface code       | `surface_49_1_7` |

The qubit order of the [[9,1,3]] rotated surface code in
`codes/surface_9_1_3.json` differs from the qubit order of MQT QECC's
`RotatedSurfaceCode(3)`. Always use the code from the JSON file with these
circuits.

## Definitions

- Fault tolerance: every set of `s <= t` faults either triggers a check (a
  detector is 1) or leaves an X error and a Z error of weight at most `s` each,
  up to stabilizers of the state.
- Detector: a `DETECTOR` instruction of the stim circuit. Its value is the
  parity of one or more measurements and is 0 if no fault occurs. A shot is
  accepted if all detectors are 0.
- Operations: CNOTs plus measurements.
- Depth: the two-qubit gate depth of the as-soon-as-possible schedule. Resets,
  single-qubit gates and measurements take no step.

## Files

- `circuits/<code>/zero_<ops>_<depth>.stim`: a circuit that prepares the logical
  all-zero state, with `<ops>` operations and depth `<depth>`. The code's qubits
  come first, in the order of the code's check matrices. The ancilla qubits
  follow. The circuits are in stim format only, because the parities of the
  `DETECTOR` instructions cannot be written in OpenQASM.
- `codes/<code>.json`: the code's check matrices `Hx` and `Hz`, its logical
  operators `Lx` and `Lz`, its length `n` and its distance `d`.
- `index.csv`: one row per circuit, with these columns:

  | Column         | Meaning                                       |
  | -------------- | --------------------------------------------- |
  | `code`         | Name of the code (directory and JSON file)    |
  | `n`, `k`, `d`  | Code parameters [[n,k,d]]                     |
  | `t`            | Number of faults the circuit tolerates        |
  | `state`        | Prepared logical state (`zero`: all-zero)     |
  | `ops`          | Operations: `cnots` plus `measurements`       |
  | `cnots`        | Two-qubit gates                               |
  | `measurements` | Measurements                                  |
  | `depth`        | Two-qubit gate depth                          |
  | `qubits`       | Qubits, ancilla qubits included               |
  | `file`         | Path of the circuit file                      |

- `simulate.py`: estimates the acceptance rate and the logical error rate of the
  circuits by Monte Carlo sampling with MQT QECC's simulator.

Every circuit passed the exact fault-tolerance check of the front search, which
enumerates every set of up to `t` faults.

## Simulating a circuit

```python
import json

import numpy as np
import stim

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis.ft_search.simulation import logical_error_rate

c = json.load(open("codes/color_17_1_5.json"))
m = {key: np.array(c[key], dtype=np.int8) for key in ("Hx", "Hz", "Lx", "Lz")}
code = CSSCode(Hx=m["Hx"], Hz=m["Hz"], Lx=m["Lx"], Lz=m["Lz"], distance=c["d"])
circuit = stim.Circuit.from_file("circuits/color_17_1_5/zero_61_8.stim")
ler, acceptance, errors, shots = logical_error_rate(circuit, code, 1e-3, min_errors=100, max_shots=10**8)
```

`logical_error_rate` returns the logical error rate per logical qubit, the
acceptance rate, the number of logical errors and the number of shots. Its
default noise model applies depolarizing noise of strength `p` after every gate
and every reset and before every measurement, and of strength `p / 100` on idle
qubits. Each reset happens as late as possible, just before the first gate on
its qubit.

To simulate all circuits of one code, run

```sh
python simulate.py --code surface_9_1_3
```

`python simulate.py --help` lists the other options.

## How the circuits were found

The circuits come from long runs of the search on several start circuits per
code. Some start circuits and ancilla preparation circuits came from an external
synthesis tool, plugged in as a further synthesis mode
(`ft_search.synth.CommandSynthesizer`). With the default settings, a run may
find different circuits.

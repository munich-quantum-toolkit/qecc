# Noise simulation

MQT QECC separates a noise simulation into three independent concepts:

1. A **channel** describes what can happen at one location.
2. A **noise model** assigns channels to physical locations.
3. A **backend adapter** applies or samples the model for a concrete simulator.

All channel and model objects are immutable and validate their parameters during
construction. This makes one configuration safe to share across experiments.

## Circuit-level Stim noise

The following model applies biased Pauli noise after single-qubit gates,
depolarizing noise after two-qubit gates and resets, measurement flips, and
weaker depolarizing noise to initialized idle qubits:

```python
import stim

from mqt.qecc.noise import (
    BitFlipChannel,
    CircuitNoiseModel,
    DepolarizingChannel,
    ParallelSchedule,
    PauliChannel,
    StimCircuitNoiseAdapter,
)

model = CircuitNoiseModel(
    single_qubit_gate=PauliChannel.from_total_probability(0.001, bias=(1, 1, 10)),
    two_qubit_gate=DepolarizingChannel(0.01),
    reset=DepolarizingChannel(0.002),
    measurement=BitFlipChannel(0.003),
    idle=DepolarizingChannel(0.0001),
)

circuit = stim.Circuit("R 0 1\nH 0\nCX 0 1\nM 0 1")
noisy_circuit = StimCircuitNoiseAdapter(
    model,
    schedule=ParallelSchedule(reset_timing="alap"),
).apply(circuit)
```

Idle noise requires a schedule because the circuit alone does not determine the
duration of idle locations. `ParallelSchedule` groups nonconflicting operations;
`SequentialSchedule` gives each target group its own time step. Without idle
noise, omit the schedule to preserve source ordering.

`PauliChannel` is a one-qubit channel. Applying it at a two-qubit location is an
error rather than an implicit approximation. Use `DepolarizingChannel` for a
uniform two-qubit Pauli channel.

The existing `CircuitLevelNoise`, `CircuitLevelNoiseIdlingParallel`, and
`CircuitLevelNoiseIdlingSequential` classes remain available as compatibility
interfaces and delegate to this adapter.

## Phenomenological noise

The NumPy backend samples data errors and noisy syndromes from an explicitly
provided random-number generator:

```python
import numpy as np

from mqt.qecc.noise import (
    GaussianReadoutChannel,
    PauliChannel,
    PhenomenologicalNoiseModel,
    PhenomenologicalNoiseSampler,
)

model = PhenomenologicalNoiseModel(
    data=PauliChannel.from_total_probability(0.01, bias=(1, 1, 5)),
    syndrome=GaussianReadoutChannel.from_bit_error_probability(0.02),
)
sampler = PhenomenologicalNoiseSampler(model, rng=np.random.default_rng(42))

x_error, z_error = sampler.sample_data(n_qubits=7)
analog_syndrome = sampler.sample_syndrome(np.array([0, 1, 0]))
```

Owning the generator makes complete simulation sequences reproducible and avoids
hidden process-global randomness. Sampled residual errors are returned as new
arrays; inputs are not mutated.

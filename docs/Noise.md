# Noise simulation

MQT QECC separates the description of noise from its execution. A channel
describes a local error, a model assigns channels to parts of an experiment, and
an adapter or sampler realizes that model for a particular simulation method.
This separation allows the same noise configuration to be inspected, validated,
and reused without embedding simulator-specific instructions in it.

All channel and model objects are immutable and validate their parameters during
construction.

## Typical data flow

Circuit-level noise follows this path:

```text
Channels → Circuit noise model → Schedule → Backend adapter → Noisy circuit
```

Phenomenological simulation does not require a circuit schedule or circuit
backend:

```text
Channels → Phenomenological noise model → Sampler → Error and syndrome samples
```

The adapter and sampler are execution boundaries. They interpret a model, but
the model itself does not depend on Stim, NumPy, or another simulator.

## Architecture layers

The shared noise functionality is constructed as an adaptable layered
architecture:

| Layer | Module                              | Main abstractions                                                                                    | Responsibility                                                                                   |
| ----: | ----------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
|     1 | {py:mod}`mqt.qecc.noise.channels`   | `IdentityChannel`, `BitFlipChannel`, `DepolarizingChannel`, `PauliChannel`, `GaussianReadoutChannel` | Local quantum errors or a classical corruption of a measurement result.                          |
|     2 | {py:mod}`mqt.qecc.noise.models`     | `CircuitNoiseModel`, `PhenomenologicalNoiseModel`                                                    | Assign discrete channels to circuit locations or to phenomenological data and syndrome noise.    |
|     3 | {py:mod}`mqt.qecc.noise.scheduling` | `ParallelSchedule`, `SequentialSchedule`                                                             | Circuit time steps used to identify idle locations.                                              |
|     4 | {py:mod}`mqt.qecc.noise.adapters`   | `StimCircuitNoiseAdapter`                                                                            | Translate a circuit model and optional schedule into the representation of a simulation backend. |
|     4 | {py:mod}`mqt.qecc.noise.sampling`   | `PhenomenologicalNoiseSampler`                                                                       | Randomly draw data and syndrome errors.                                                          |

The layers describe a *uses* relationship rather than class inheritance.
Scheduling applies only to circuit-level models, while direct sampling applies
to phenomenological models.

```{note}
MQT [YAQS](https://github.com/munich-quantum-toolkit/yaqs) could be potentially
integrated through an additional adapter in `mqt.qecc.noise.adapters`,
describing continuous-time open-system processes.
```

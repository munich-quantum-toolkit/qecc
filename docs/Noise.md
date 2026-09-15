# Noise Simulation

MQT QECC separates the description of noise from its execution. A channel
describes a local error, a location-aware model assigns channels to parts of an
experiment, and an adapter or sampler realizes that model for a particular
simulation method. This separation allows the same noise configuration to be
inspected, validated, and reused without embedding simulator-specific
instructions in it.

All channel and model objects are immutable and validate their parameters during
construction.

## Architecture and Data Flow

The shared noise functionality is constructed as an adaptable layered
architecture, which describe a *uses* relationship rather than class
inheritance:

::::{grid} 1
:gutter: 2

:::{grid-item-card} Single-site noise channels
:text-align: center

:::

::::

::::{grid} 2 2 2 2
:gutter: 3

:::{grid-item}
:class: sd-text-center sd-fs-5

↓
:::

:::{grid-item}
:class: sd-text-center sd-fs-5

↓
:::

::::

::::{grid} 2 2 2 2
:gutter: 3

:::{grid-item-card} `CircuitNoiseModel`
:text-align: center

Assigns channels to gate, reset, measurement, and idle locations.
:::

:::{grid-item-card} `PhenomenologicalNoiseModel`
:text-align: center

Assigns channels to data qubits and syndrome measurements.
:::

::::

::::{grid} 2 2 2 2
:gutter: 3

:::{grid-item}
:class: sd-text-center sd-fs-5

↓
:::

:::{grid-item}
:class: sd-text-center sd-fs-5

↓
:::

::::

::::{grid} 2 2 2 2
:gutter: 3

:::{grid-item-card} `Schedule` + `StimCircuitNoiseAdapter`
:text-align: center
:columns: 6

Identifies circuit time steps and translates channels into Stim instructions.
:::

:::{grid-item-card} `PhenomenologicalNoiseSampler`
:text-align: center
:columns: 3

Resolves per-qubit assignments and samples data and readout noise.
:::

:::{grid-item-card} `PhenomenologicalStimAdapter`
:text-align: center
:columns: 3

Emits data and readout noise into a Stim circuit as it is built.
:::

::::

::::{grid} 2 2 2 2
:gutter: 3

:::{grid-item}
:class: sd-text-center sd-fs-5

↓
:::

:::{grid-item}
:class: sd-text-center sd-fs-5

↓
:::

::::

::::{grid} 2 2 2 2
:gutter: 3

:::{grid-item}
:class: sd-text-center

**Noisy Stim circuit**
:::

:::{grid-item}
:class: sd-text-center

**Data errors and noisy syndromes, or a noisy Stim circuit**
:::

::::

```{note}
[MQT YAQS](https://github.com/munich-quantum-toolkit/yaqs) could be potentially
integrated through an additional adapter in `mqt.qecc.noise.adapters`,
describing continuous-time open-system processes.
```

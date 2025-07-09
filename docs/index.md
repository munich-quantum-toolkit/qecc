# Welcome to QECC's documentation

QECC is a tool for quantum error correcting codes developed as part of the [Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io) by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de).

We recommend you to start with the {doc}`installation instructions <Installation>`.
Then proceed to read the {doc}`reference documentation <api/mqt/qecc/index>`.
If you are interested in the theory behind QECC, have a look at the publications in the {doc}`publication list <Publications>`.

We appreciate any feedback and contributions to the project. If you want to contribute, you can find more information in
the {doc}`Contribution <Contributing>` guide. If you are having trouble with the installation or the usage of QECC,
please let us know at our {doc}`Support <Support>` page or by reaching out to us at
[quantum.cda@xcit.tum.de](mailto:quantum.cda@xcit.tum.de).

---

```{toctree}
:hidden:

self
```

```{toctree}
:maxdepth: 1
:caption: User Guide

Installation
LightsOutDecoder
StatePrep
CatStates
Encoders
AnalogInfo
Publications
```

```{toctree}
:caption: Developers
:glob:
:maxdepth: 1

Contributing
DevelopmentGuide
Support
```

```{toctree}
:maxdepth: 3
:caption: API Reference

api/mqt/qecc/index
```

```{only} html
## Contributors and Supporters
The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/)
and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem,
which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.
<div style="margin-top: 0.5em">
<div class="only-light" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Banner">
</div>
<div class="only-dark" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%" alt="MQT Banner">
</div>
</div>
Thank you to all the contributors who have helped make MQT QECC a reality!
<p align="center">
<a href="https://github.com/cda-tum/qecc/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=cda-tum/qecc" />
</a>
</p>
```

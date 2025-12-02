[![PyPI](https://img.shields.io/pypi/v/mqt.qecc?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.qecc/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos%20%7C%20windows-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/qecc/ci.yml?branch=main&style=flat-square&logo=github&label=ci)](https://github.com/munich-quantum-toolkit/qecc/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/qecc/cd.yml?style=flat-square&logo=github&label=cd)](https://github.com/munich-quantum-toolkit/qecc/actions/workflows/cd.yml)
[![Documentation](https://img.shields.io/readthedocs/qecc?logo=readthedocs&style=flat-square)](https://mqt.readthedocs.io/projects/qecc)
[![codecov](https://img.shields.io/codecov/c/github/munich-quantum-toolkit/qecc?style=flat-square&logo=codecov)](https://codecov.io/gh/munich-quantum-toolkit/qecc)

<p align="center">
  <a href="https://mqt.readthedocs.io">
   <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-dark.svg" width="60%">
      <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-light.svg" width="60%" alt="MQT Logo">
   </picture>
  </a>
</p>

# MQT QECC - A tool for Quantum Error Correcting Codes

MQT QECC is a tool for quantum error correcting codes and numerical simulations.
It is part of the [_Munich Quantum Toolkit (MQT)_](https://mqt.readthedocs.io).

<p align="center">
  <a href="https://mqt.readthedocs.io/projects/qecc">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

## Key Features

- Decode (triangular) color codes and conduct respective numerical simulations.
  - The decoder is based on an analogy to the classical LightsOut puzzle and formulated as a MaxSAT problem.
    The SMT solver Z3 is used to determine minimal solutions of the MaxSAT problem, resulting in minimum-weight decoding estimates.
- Decode bosonic quantum LDPC codes and conduct numerical simulations for analog information decoding under phenomenological (cat qubit) noise.
- Synthesize non-deterministic and deterministic fault-tolerant state preparation circuits for qubit CSS codes.

> [!NOTE]
> Usage for _Lattice Surgery Compilation Beyond the Surface Code_ as well as _Exploiting Movable Logical Qubits for Lattice Surgery Compilation_ is described in [`docs/cococo.md`](https://github.com/munich-quantum-toolkit/qecc/blob/cococo/docs/cococo.md) in the `cococo` branch.
> The code quality in the branch is actively being improved.

> [!WARNING]
> The C++ implementation of the [union find decoder for LDPC codes](https://arxiv.org/pdf/2301.05731) and the [circuit transpilation framework](https://arxiv.org/abs/2209.0118) have been removed with `v2.0.0` and are no longer available.
> QECC is now entirely a Python package.
> For up-to-date software for decoding LDPC codes we refer to [quantumgizmos/ldpc](https://github.com/quantumgizmos/ldpc).
> If you would still like to use these features, they are available in `mqt.qecc` versions `v2.0.0`.

If you have any questions, feel free to create a [discussion](https://github.com/munich-quantum-toolkit/qecc/discussions) or an [issue](https://github.com/munich-quantum-toolkit/qecc/issues) on [GitHub](https://github.com/munich-quantum-toolkit/qecc).

## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Partner Logos">
  </picture>
</p>

Thank you to all the contributors who have helped make MQT QECC a reality!

<p align="center">
<a href="https://github.com/munich-quantum-toolkit/qecc/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/qecc" />
</a>
</p>

The MQT will remain free, open-source, and permissively licensed—now and in the future.
We are firmly committed to keeping it open and actively maintained for the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories: https://github.com/munich-quantum-toolkit
- Contributing code, documentation, tests, or examples via issues and pull requests
- Citing the MQT in your publications (see [Cite This](#cite-this))
- Citing our research in your publications (see [References](https://mqt.readthedocs.io/projects/qecc/en/latest/references.html))
- Using the MQT in research and teaching, and sharing feedback and use cases

## Getting Started

`mqt.qecc` is available via [PyPI](https://pypi.org/project/mqt.qecc/).

```console
(venv) $ pip install mqt.qecc
```

**Detailed documentation and examples are available at [ReadTheDocs](https://mqt.readthedocs.io/projects/qecc).**

## System Requirements

MQT QECC can be installed on all major operating systems with all [officially supported Python versions](https://devguide.python.org/versions/).
Building (and running) is continuously tested under Linux, macOS, and Windows using the [latest available system versions for GitHub Actions](https://github.com/actions/runner-images).

## Cite This

Please cite the work that best fits your use case.

### The Munich Quantum Toolkit (the project)

When discussing the overall MQT project or its ecosystem, cite the MQT Handbook:

```bibtex
@inproceedings{mqt,
  title        = {The {{MQT}} Handbook: {{A}} Summary of Design Automation Tools and Software for Quantum Computing},
  shorttitle   = {{The MQT Handbook}},
  author       = {Wille, Robert and Berent, Lucas and Forster, Tobias and Kunasaikaran, Jagatheesan and Mato, Kevin and Peham, Tom and Quetschlich, Nils and Rovara, Damian and Sander, Aaron and Schmid, Ludwig and Schoenberger, Daniel and Stade, Yannick and Burgholzer, Lukas},
  year         = 2024,
  booktitle    = {IEEE International Conference on Quantum Software (QSW)},
  doi          = {10.1109/QSW62656.2024.00013},
  eprint       = {2405.17543},
  eprinttype   = {arxiv},
  addendum     = {A live version of this document is available at \url{https://mqt.readthedocs.io}}
}
```

### Peer-Reviewed Research

When citing the underlying methods and research, please reference the most relevant peer-reviewed publications from the list below:

[[1]](https://arxiv.org/pdf/2501.05527)
L. Schmid, T.Peham, L. Berent, M. Müller, and R. Wille.
Deterministic Fault-Tolerant State Preparation for Near-Term Quantum Error Correction: Automatic Synthesis Using Boolean Satisfiability

[[2]](https://arxiv.org/pdf/2408.11894)
T. Peham, L. Schmid, L. Berent, M. Müller, and R. Wille.
Automated Synthesis of Fault-Tolerant State Preparation Circuits for Quantum Error Correction Codes
_PRX Quantum 6, 020330_, 2025.

[[3]](https://arxiv.org/pdf/2311.01328)
L. Berent, T. Hillmann, J. Eisert, R. Wille, and J. Roffe.
Analog information decoding of bosonic quantum LDPC codes.
_PRX Quantum 5, 020349_, 2024.

[[4]](https://arxiv.org/pdf/2303.14237)
L. Berent, L. Burgholzer, P. J. Derks, J. Eisert, and R. Wille.
Decoding quantum color codes with MaxSAT.
_Quantum 8, 1506_, 2024.

[[5]](https://arxiv.org/pdf/2301.05731)
T. Grurl, C. Pichler, J. Fuss, and R. Wille.
Automatic Implementation and Evaluation of Error-Correcting Codes for Quantum Computing: An Open-Source Framework for Quantum Error-Correction.
_International Conference on VLSI Design and International Conference on Embedded Systems (VLSID)_, 2023.

[[6]](https://arxiv.org/pdf/2209.01180)
L. Berent, L. Burgholzer, and R. Wille.
Software Tools for Decoding Quantum Low-Density Parity Check Codes.
_Asia and South Pacific Design Automation Conference (ASP-DAC)_, 2023.

---

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 101001318), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as well as the Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>

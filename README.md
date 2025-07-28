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

# MQT QECC: A tool for Quantum Error Correcting Codes

A tool for quantum error correcting codes and numerical simulations developed as part of the [_Munich Quantum Toolkit (MQT)_](https://mqt.readthedocs.io) by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/).

The tool can be used to:

- Decode (triangular) color codes and conduct respective numerical simulations.
  - The decoder is based on an analogy to the classical LightsOut puzzle and formulated as a MaxSAT problem. The SMT solver
    Z3 is used to determine minimal solutions of the MaxSAT problem, resulting in minimum-weight decoding estimates.
- Decode bosonic quantum LDPC codes and conduct numerical simulations for analog information decoding under phenomenological (cat qubit) noise.
- Synthesize non-deterministic and deterministic fault-tolerant state preparation circuits for qubit CSS codes.

<p align="center">
  <a href="https://mqt.readthedocs.io/projects/qecc">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

> [!WARNING]
> The C++ implementation of the [union find decoder for LDPC codes](https://arxiv.org/pdf/2301.05731) and the [circuit transpilation framework](https://arxiv.org/abs/2209.0118) have been removed with `v2.0.0` and are no longer available.
> QECC is now entirely a Python package.
> For up to date software for decoding LDPC codes we refer to [quantumgizmos/ldpc](https://github.com/quantumgizmos/ldpc).

If you would like to use these features, they are available in `mqt.qecc` version <2.0.0.

Basic usage for _lattice surgery compilation beyond the surface code_ is described [here](https://github.com/munich-quantum-toolkit/qecc/blob/ls-compilation/docs/Co3.rst) in the branch `ls-compilation` whose code quality improvements are work in progress.

If you have any questions, feel free to contact us via [quantum.cda@xcit.tum.de](mailto:quantum.cda@xcit.tum.de) or by
creating an issue on [GitHub](https://github.com/munich-quantum-toolkit/qecc/issues).

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

## Getting Started

`mqt.qecc` is available via [PyPI](https://pypi.org/project/mqt.qecc/) for Linux, macOS, as well as Windows and supports Python 3.9 to 3.13.

```console
(venv) $ pip install mqt.qecc
```

**Detailed documentation on all available methods, options, and input formats is available
at [ReadTheDocs](https://mqt.readthedocs.io/projects/qecc).**

## Reference

If you use our tool for your research, we will be thankful if you refer to it by citing the appropriate publication:

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2311.01328&color=inactive&style=flat-square)](https://arxiv.org/abs/2501.05527)
  L. Schmid, T.Peham, L. Berent, M. Müller, and R. Wille, "Deterministic Fault-Tolerant State Preparation for Near-Term Quantum Error Correction: Automatic Synthesis Using Boolean Satisfiability".

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2311.01328&color=inactive&style=flat-square)](https://arxiv.org/abs/2408.11894)
  T. Peham, L. Schmid, L. Berent, M. Müller, and R. Wille, "Automated Synthesis of Fault-Tolerant State Preparation Circuits for Quantum Error Correction Codes".

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2311.01328&color=inactive&style=flat-square)](https://arxiv.org/abs/2311.01328)
  L. Berent, T. Hillmann, J. Eisert, R. Wille, and J. Roffe, "Analog information decoding of bosonic quantum LDPC codes".

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2303.14237&color=inactive&style=flat-square)](https://arxiv.org/abs/2303.14237)
  L. Berent, L. Burgholzer, P.J. Derks, J. Eisert, and R. Wille, "Decoding quantum color codes with MaxSAT".

  The dataset used in the paper evaluation on decoding quantum color codes is available on Zenodo:
  [![a](https://img.shields.io/static/v1?label=DOI&message=10.5281/zenodo.7760135&color=inactive&style=flat-square)](https://doi.org/10.5281/zenodo.7760135)

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2301.05731&color=inactive&style=flat-square)](https://arxiv.org/pdf/2301.05731)
  T. Grurl, C. Pichler, J. Fuss and R. Wille, "Automatic Implementation and Evaluation of Error-Correcting Codes for
  Quantum Computing: An Open-Source Framework for Quantum Error-Correction," in International Conference on VLSI
  Design and International Conference on Embedded Systems (VLSID), 2023

- [![a](https://img.shields.io/static/v1?label=arXiv&message=2209.01180&color=inactive&style=flat-square)](https://arxiv.org/abs/2209.01180)
  L. Berent, L. Burgholzer, and R.
  Wille, "[Software Tools for Decoding Quantum Low-Density Parity Check Codes](https://arxiv.org/abs/2209.01180),"
  in Asia and South Pacific Design Automation Conference (ASP-DAC), 2023

---

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European
Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement
No. 101001318), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as well as the
Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>

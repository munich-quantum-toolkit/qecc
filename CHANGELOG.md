<!-- Entries in each category are sorted by merge time, with the latest PRs appearing first. -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on a mixture of [Keep a Changelog] and [Common Changelog].
This project adheres to [Semantic Versioning], with the exception that minor releases may include breaking changes.

## [Unreleased]

### Added

- Added `gottesman_encoding_circuit` methods that constructs a stim encoding circuit for a given stabilizer code using the method described in Gottesman's "Surviving as a Quantum Computer in a Classical World" Chapter 6.4.1. ([#486]) ([**@pehamtom**])
- Added class `SteaneNDFTStatePrepSimulator` for simulating non-deterministic state preparation protocols for CSS codes using verification with multiple ancilla states. ([#462]) ([**@pehamtom**])
- Extended estimation of error rates in `NoisyNDFTStatePrepSimulator` via `secondary_logical_error_rate`. Now Z (X) error rates can also be estimated for the preparation of logical zero (plus). ([#462]) ([**@pehamtom**])
- Added `ComposedNoiseModel` class that allows for composition of noise models. )([#462]) ([**@pehamtom**])
- Added functionality to concatenate stim circuits along specific qubits. Add functionality to concatenate stim circuits along specific qubits ([#461]) ([**@pehamtom**])
- Added `NoiseModel` class for applying noise to a given stim circuit. ([#453]) ([**@pehamtom**])
- New `PureFaultSet` class for representing collections of X or Z faults. ([#443]) ([**@pehamtom**])
- New `CNOTCircuit` class to serve as an intermediate representation during circuit synthesis for simplifying work with CSS encoding isometries. ([#443]) ([**@pehamtom**])
- Combinatorial search methods for constructing fault-tolerant cat state preparation circuits. ([#543]) ([**@pehamtom**])

### Changed

- Stop testing on x86 macOS systems ([#592]) ([**@denialhaag**])
- Move Python tests from `test/python` to `tests`. ([#482]) ([**@denialhaag**])
- `NoisyNDFTStatePrepSimulator` simulates generalized post-selection based state preparation protocols. Old functionality for simulating state preparation protocols post-selected on stabilizer measurements can be found in the class `VerificationNDFTStatePrepSimulator`. ([#462]) ([**@pehamtom**])
- Refactored state preparation circuit synthesis code to utilize the new `PureFaultSet` and `CNOTCircuit` classes. ([#443]) ([**@pehamtom**])
- Refactored encoding circuit synthesis code to utilize the new `PureFaultSet` and `CNOTCircuit` classes. ([#443]) ([**@pehamtom**])
- Renamed `StatePrepCircuit` class to `FaultyStatePrepCircuit`, reflecting its new role in combining circuit and fault information. ([#443]) ([**@pehamtom**])
- Changed the construction in `CatStatePreparationExperiment` to allow for ancillas with less qubits than the data cat state.

### Removed

- Drop support for Python 3.9 ([#503]) ([**@denialhaag**])

## [1.9.0] - 2025-03-14

_📚 Refer to the [GitHub Release Notes](https://github.com/munich-quantum-toolkit/qecc/releases) for previous changelogs._

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/qecc/compare/v1.9.0...HEAD
[1.9.0]: https://github.com/munich-quantum-toolkit/qecc/releases/tag/v1.9.0

<!-- PR links -->

[#592]: https://github.com/munich-quantum-toolkit/qecc/pull/592
[#543]: https://github.com/munich-quantum-toolkit/qecc/pull/543
[#503]: https://github.com/munich-quantum-toolkit/qecc/pull/503
[#499]: https://github.com/munich-quantum-toolkit/qecc/pull/499
[#486]: https://github.com/munich-quantum-toolkit/qecc/pull/486
[#482]: https://github.com/munich-quantum-toolkit/qecc/pull/482
[#462]: https://github.com/munich-quantum-toolkit/qecc/pull/462
[#461]: https://github.com/munich-quantum-toolkit/qecc/pull/461
[#453]: https://github.com/munich-quantum-toolkit/qecc/pull/453
[#443]: https://github.com/munich-quantum-toolkit/qecc/pull/443

<!-- Contributor -->

[**@pehamtom**]: https://github.com/pehamtom
[**@denialhaag**]: https://github.com/denialhaag

<!-- General links -->

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Common Changelog]: https://common-changelog.org
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[GitHub Release Notes]: https://github.com/munich-quantum-toolkit/qecc/releases

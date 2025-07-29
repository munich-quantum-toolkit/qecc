<!-- Entries in each category are sorted by merge time, with the latest PRs appearing first. -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on a mixture of [Keep a Changelog] and [Common Changelog].
This project adheres to [Semantic Versioning], with the exception that minor releases may include breaking changes.

## [Unreleased]

### Added

- New `PureFaultSet` class for representing collections of X or Z faults. ([#443]) ([**@pehamtom**])
- New `CNOTCircuit` class to serve as an intermediate representation during circuit synthesis for simplifying work with CSS encoding isometries. ([#443]) ([**@pehamtom**])
- Added `NoiseModel` class for applying noise to a given stim circuit. ([#453]) ([**@pehamtom**])
- Added functionality to concatenate stim circuits along specific qubits.

### Changed

- Refactored state preparation circuit synthesis code to utilize the new `PureFaultSet` and `CNOTCircuit` classes. ([#443]) ([**@pehamtom**])
- Refactored encoding circuit synthesis code to utilize the new `PureFaultSet` and `CNOTCircuit` classes. ([#443]) ([**@pehamtom**])
- Renamed `StatePrepCircuit` class to `FaultyStatePrepCircuit`, reflecting its new role in combining circuit and fault information. ([#443]) ([**@pehamtom**])

## [1.9.0] - 2025-03-14

_📚 Refer to the [GitHub Release Notes](https://github.com/munich-quantum-toolkit/qecc/releases) for previous changelogs._

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/qecc/compare/v1.9.0...HEAD
[1.9.0]: https://github.com/munich-quantum-toolkit/qecc/releases/tag/v1.9.0

<!-- PR links -->

[#443]: https://github.com/munich-quantum-toolkit/qecc/pull/443
[#453]: https://github.com/munich-quantum-toolkit/qecc/pull/453
[#461]: https://github.com/munich-quantum-toolkit/qecc/pull/461

<!-- Contributor -->

<!-- General links -->

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Common Changelog]: https://common-changelog.org
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[GitHub Release Notes]: https://github.com/munich-quantum-toolkit/qecc/releases

# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

### End of support for x86 macOS systems

Starting with this release, MQT QECC no longer supports x86 macOS systems.
This comes as a result of GitHub removing the `macos-13` runners from their infrastructure.
Users on x86 macOS systems can still install MQT QECC.
However, these systems are no longer tested in our CI and we can no longer guarantee that MQT QECC builds and runs correctly.

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/qecc/compare/v1.9.0...HEAD

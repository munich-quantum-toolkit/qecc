# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

### End of support for Python 3.9

Starting with this release, MQT QECC no longer supports Python 3.9.
This is in line with the scheduled end of life of the version.
As a result, MQT QECC is no longer tested under Python 3.9 and requires Python 3.10 or later.

### End of support for x86 macOS systems

Starting with this release, we can no longer guarantee support for x86 macOS systems.
x86 macOS systems are no longer tested in our CI and we can no longer guarantee that MQT QECC installs and runs correctly on them.

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/qecc/compare/v1.9.0...HEAD

<!--- This file has been generated from an external template. Please do not modify it directly. -->
<!--- Changes should be contributed to https://github.com/munich-quantum-toolkit/templates. -->

# MQT QECC

## Python

- Install package: `uv sync`
- Run tests: `uv run pytest`
- Nox test shortcuts: `uvx nox -s tests`, `uvx nox -s minimums`
- Python 3.14 variants: `uvx nox -s tests-3.14`, `uvx nox -s minimums-3.14`

## Documentation

- Sources: `docs/`
- Build docs locally: `uvx nox --non-interactive -s docs`
- Link check: `uvx nox -s docs -- -b linkcheck`

## Tech Stack

### General

- `prek` for pre-commit hooks

### Python

- Python 3.10+
- `uv` for installation, packaging, and tooling
- `ruff` for formatting/linting (configured in `pyproject.toml`)
- `ty` for type checking
- `pytest` for unit tests (located in `test/python/`)
- `nox` for task orchestration (tests, linting, docs)

### Documentation

- `sphinx`
- MyST (Markdown)
- Furo theme

## Development Guidelines

### General

- MUST run `uvx nox -s lint` after every batch of changes.
  This runs the full `prek` hook set from `.pre-commit-config.yaml` (including `ruff`, `typos`, `ty`, formatting, and metadata checks).
  All hooks must pass before submitting.
- MUST add or update tests for every code change, even if not explicitly requested.
- MUST follow existing code style by checking neighboring files for patterns.
- MUST update `CHANGELOG.md` and `UPGRADING.md` when changes are user-facing, breaking, or otherwise noteworthy.
- MUST include a commit footer attribution in the form `Assisted-by: [Model Name] via [Tool Name]` (example: `Assisted-by: Claude Sonnet 4.6 via GitHub Copilot`) if AI tools are used to prepare a commit.
- NEVER modify files that start with "This file has been generated from an external template. Please do not modify it directly."
  These files are managed by [the MQT templates action](https://github.com/munich-quantum-toolkit/templates) and changes will be overwritten.
- PREFER running targeted tests over the full test suite during development.

### Python

- MUST use Google-style docstrings
- PREFER running a single Python version over the full test suite during development.
- PREFER fixing reported warnings over suppressing them (e.g., with `# noqa` comments for ruff); only add ignore rules when necessary and document why.
- PREFER fixing typing issues reported by `ty` before adding suppression comments (`# ty: ignore[code]`); suppressions are sometimes necessary for incompletely typed libraries (e.g., Qiskit).

## Self-Review Checklist

- Did `uvx nox -s lint` pass without errors?
- Are all changes covered by at least one automated test?
- Are `CHANGELOG.md` and `UPGRADING.md` updated when changes are user-facing, breaking, or otherwise noteworthy?

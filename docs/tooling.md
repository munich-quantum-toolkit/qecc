<!--- This file has been generated from an external template. Please do not modify it directly. -->
<!--- Changes should be contributed to https://github.com/munich-quantum-toolkit/templates. -->

# Tooling

This page summarizes the main tools, software, and standards used in MQT QECC.
It serves as a quick reference for new contributors and users who want to understand the project's ecosystem.

## Python

| Tool       | Description                                                                              | Links / Notes                                                                                                                                          |
| ---------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **uv**     | Fast Python package and project manager (install, venv, dependencies).                   | [Documentation](https://docs.astral.sh/uv/). Recommended over {code}`pip` for installs and development.                                                |
| **nox**    | Task automation for tests, lint, docs, and other sessions defined in {code}`noxfile.py`. | [Documentation](https://nox.thea.codes/en/stable/). Run sessions with {code}`nox -s <session>`.                                                        |
| **prek**   | Runs hooks (formatting, linting) before each commit.                                     | [Documentation](https://prek.j178.dev). Install and run via {code}`prek install`, {code}`prek run` (staged files), or {code}`prek run -a` (all files). |
| **ruff**   | Linter and formatter for Python, written in Rust.                                        | [Documentation](https://docs.astral.sh/ruff/). Used in prek and CI.                                                                                    |
| **ty**     | Fast Python type checker and language server.                                            | [Documentation](https://docs.astral.sh/ty/).                                                                                                           |
| **pytest** | Testing framework for Python.                                                            | [Documentation](https://docs.pytest.org/). Run via {code}`nox -s tests` or {code}`uv run pytest`.                                                      |

The project adheres to modern standards and practices. For the Python ecosystem, we make use of the following standards:

| Standard    | Description                                                     | Links / Notes                                       |
| ----------- | --------------------------------------------------------------- | --------------------------------------------------- |
| **PEP 8**   | Style guide for Python code.                                    | [Documentation](https://peps.python.org/pep-0008/). |
| **PEP 518** | Specifying build system requirements in {code}`pyproject.toml`. | [Documentation](https://peps.python.org/pep-0518/). |
| **PEP 621** | Storing project metadata in {code}`pyproject.toml`.             | [Documentation](https://peps.python.org/pep-0621/). |
| **PEP 639** | Standardized license metadata in {code}`pyproject.toml`.        | [Documentation](https://peps.python.org/pep-0639/). |
| **PEP 723** | Inline script metadata for efficient script execution.          | [Documentation](https://peps.python.org/pep-0723/). |
| **PEP 735** | Dependency groups in {code}`pyproject.toml`.                    | [Documentation](https://peps.python.org/pep-0735/). |

## Documentation

| Tool       | Description                                | Links / Notes                                                                               |
| ---------- | ------------------------------------------ | ------------------------------------------------------------------------------------------- |
| **Sphinx** | Documentation generator.                   | [Documentation](https://www.sphinx-doc.org/). Docs source in {code}`docs/`.                 |
| **MyST**   | Markdown flavor for Sphinx (used in docs). | [Documentation](https://myst-parser.readthedocs.io/). Enables rich Markdown in doc sources. |

## CI and Quality

| Tool               | Description                                 | Links / Notes                                                          |
| ------------------ | ------------------------------------------- | ---------------------------------------------------------------------- |
| **GitHub Actions** | CI workflows (build, test, lint, coverage). | [Reusable MQT Workflows] in {code}`.github/workflows/`; see [Actions]. |
| **Codecov**        | Code coverage reporting.                    | [Codecov] for this repo.                                               |
| **CodeRabbit**     | Automated code review on PRs.               | [CodeRabbit](https://www.coderabbit.ai/). See {doc}`contributing`.     |
| **pre-commit.ci**  | Runs pre-commit in CI and can auto-fix.     | [pre-commit.ci](https://pre-commit.ci).                                |

[Actions]: https://github.com/munich-quantum-toolkit/qecc/actions
[Codecov]: https://codecov.io/gh/munich-quantum-toolkit/qecc
[Reusable MQT Workflows]: https://github.com/munich-quantum-toolkit/workflows

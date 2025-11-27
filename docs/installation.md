<!--- This file has been generated from an external template. Please do not modify it directly. -->
<!--- Changes should be contributed to https://github.com/munich-quantum-toolkit/templates. -->

# Installation

MQT QECC is a Python package available on [PyPI](https://pypi.org/project/mqt.qecc/).
It can be installed on all major operating systems with all [officially supported Python versions](https://devguide.python.org/versions/).

:::::{tip}
:name: uv-recommendation

We recommend using [{code}`uv`][uv].
It is a fast Python package and project manager by [Astral](https://astral.sh/) (creators of [{code}`ruff`][ruff]).
It can replace {code}`pip` and {code}`virtualenv`, automatically manages virtual environments, installs packages, and can install Python itself.
It is significantly faster than {code}`pip`.

If you do not have {code}`uv` installed, install it with:

::::{tab-set}
:::{tab-item} macOS and Linux

```console
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

:::
:::{tab-item} Windows

```console
$ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

:::
::::

See the [uv documentation][uv] for more information.

:::::

::::{tab-set}
:sync-group: installer

:::{tab-item} {code}`uv` _(recommended)_
:sync: uv

```console
$ uv pip install mqt.qecc
```

:::

:::{tab-item} {code}`pip`
:sync: pip

```console
(.venv) $ python -m pip install mqt.qecc
```

:::
::::

Verify the installation:

```console
(.venv) $ python -c "import mqt.qecc; print(mqt.qecc.__version__)"
```

This prints the installed package version.

## Integrating MQT QECC into Your Project

To use the MQT QECC Python package in your project, add it as a dependency in your {code}`pyproject.toml` or {code}`setup.py`.
This ensures the package is installed when your project is installed.

::::{tab-set}

:::{tab-item} {code}`uv` _(recommended)_

```console
$ uv add mqt.qecc
```

:::

:::{tab-item} {code}`pyproject.toml`

```toml
[project]
# ...
dependencies = ["mqt.qecc>=<version>"]
# ...
```

:::

:::{tab-item} {code}`setup.py`

```python
from setuptools import setup

setup(
    # ...
    install_requires=["mqt.qecc>=<version>"],
    # ...
)
```

:::
::::

(development-setup)=

## Development Setup

Set up a reproducible development environment for MQT QECC.
This is the recommended starting point for both bug fixes and new features.
For detailed guidelines and workflows, see {doc}`contributing`.

1.  Get the code:

    ::::{tab-set}
    :::{tab-item} External Contribution
    If you do not have write access to the [munich-quantum-toolkit/qecc](https://github.com/munich-quantum-toolkit/qecc) repository, fork the repository on GitHub (see <https://docs.github.com/en/get-started/quickstart/fork-a-repo>) and clone your fork locally.

    ```console
    $ git clone git@github.com:your_name_here/qecc.git mqt-qecc
    ```

    :::
    :::{tab-item} Internal Contribution
    If you have write access to the [munich-quantum-toolkit/qecc](https://github.com/munich-quantum-toolkit/qecc) repository, clone the repository locally.

    ```console
    $ git clone git@github.com/munich-quantum-toolkit/qecc.git mqt-qecc
    ```

    :::
    ::::

2.  Change into the project directory:

    ```console
    $ cd mqt-qecc
    ```

3.  Create a branch for local development:

    ```console
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

4.  Install development tools:

    We highly recommend using modern, fast tooling for the development workflow.
    We recommend using [{code}`uv`][uv].
    If you don't have {code}`uv`, follow the installation instructions in the recommendation above (see {ref}`tip above <uv-recommendation>`).
    See the [uv documentation][uv] for more information.

    We also recommend installing [{code}`pre-commit`][pre-commit] to automatically run checks before each commit and [{code}`nox`][nox] to automate common development tasks.

    ::::{tab-set}
    :sync-group: installer

    :::{tab-item} {code}`uv` _(recommended)_
    :sync: uv
    The easiest way to install {code}`pre-commit` and {code}`nox` is via [{code}`uv`][uv]:

    ```console
    $ uv tool install pre-commit
    $ uv tool install nox
    ```

    :::
    :::{tab-item} {code}`brew`
    :sync: brew
    On macOS with Homebrew, you can install {code}`pre-commit` and {code}`nox` with:

    ```console
    $ brew install pre-commit nox
    ```

    :::
    :::{tab-item} {code}`pipx`
    :sync: pipx
    If you prefer to use [{code}`pipx`][pipx], you can install {code}`pre-commit` and {code}`nox` with:

    ```console
    $ pipx install pre-commit
    $ pipx install nox
    ```

    :::
    :::{tab-item} {code}`pip`
    :sync: pip
    If you prefer to use regular {code}`pip` (preferably in a virtual environment), you can install {code}`pre-commit` and {code}`nox` with:

    ```console
    $ pip install pre-commit nox
    ```

    :::
    ::::

    Then enable the {code}`pre-commit` hooks with:

    ```console
    $ pre-commit install
    ```

<!-- Links -->

[FetchContent]: https://cmake.org/cmake/help/latest/module/FetchContent.html
[git-submodule]: https://git-scm.com/docs/git-submodule
[nox]: https://nox.thea.codes/en/stable/
[pipx]: https://pypa.github.io/pipx/
[pre-commit]: https://pre-commit.com/
[ruff]: https://docs.astral.sh/ruff/
[uv]: https://docs.astral.sh/uv/

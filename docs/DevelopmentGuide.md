# Development Guide

Ready to contribute to the project? This guide will get you started.

## Initial Setup

1. Get the code

   ::::{tab-set}
   :::{tab-item} External Contribution
   If you do not have write access to the [munich-quantum-toolkit/qecc](https://github.com/munich-quantum-toolkit/qecc) repository,
   fork the repository on GitHub (see <https://docs.github.com/en/get-started/quickstart/fork-a-repo>)
   and clone your fork locally.

   ```console
   $ git clone git@github.com:your_name_here/mqt-qecc.git
   ```

   :::
   :::{tab-item} Internal Contribution
   If you do have write access to the [munich-quantum-toolkit/qecc](https://github.com/munich-quantum-toolkit/qecc) repository,
   clone the repository locally.

   ```console
   $ git clone git@github.com/munich-quantum-toolkit/qecc.git
   ```

   :::
   ::::

2. Change into the project directory

   ```console
   $ cd mqt-qecc
   ```

3. Create a branch for local development

   ```console
   $ git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

4. We highly recommend using [`uv`](https://docs.astral.sh/uv/).
   It is an extremely fast Python package and project manager, written in Rust and developed by [Astral](https://astral.sh/) (the same team behind [`ruff`](https://docs.astral.sh/ruff/)).
   It can act as a drop-in replacement for `pip` and `virtualenv`, and provides a more modern and faster alternative to the traditional Python package management tools.
   It automatically handles the creation of virtual environments and the installation of packages, and is much faster than `pip`.
   Additionally, it can even set up Python for you if it is not installed yet.

   If you do not have `uv` installed yet, you can install it via:

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
   Check out their excellent [documentation](https://docs.astral.sh/uv/) for more information.

5. We also highly recommend to install and set up [pre-commit](https://pre-commit.com/) to automatically
   run a set of checks before each commit.

   ::::{tab-set}
   :::{tab-item} via `uv`
   :sync: uv
   The easiest way to install pre-commit is via [uv](https://docs.astral.sh/uv/).

   ```console
   $ uv tool install pre-commit
   ```

   :::
   :::{tab-item} via `brew`
   :sync: brew
   If you use macOS, then pre-commit is in Homebrew, and you can use

   ```console
   $ brew install pre-commit
   ```

   :::
   :::{tab-item} via `pipx`
   :sync: pipx
   If you prefer to use [pipx](https://pypa.github.io/pipx/), you can install pre-commit with

   ```console
   $ pipx install pre-commit
   ```

   :::
   :::{tab-item} via `pip`
   :sync: pip
   If you prefer to use regular `pip` (preferably in a virtual environment), you can install pre-commit with

   ```console
   $ pip install pre-commit
   ```

   :::
   ::::

   Afterwards, you can install the pre-commit hooks with

   ```console
   $ pre-commit install
   ```

## Working on the package

The Python package lives in the {code}`src/mqt/qecc` directory.

::::::{tab-set}
:sync-group: installer

:::::{tab-item} uv _(recommended)_
:sync: uv
Getting the project up and running locally using `uv` is as simple as running:

```console
$ uv sync
```

This will

- download a suitable version of Python for you (if you don't have it installed yet),
- create a virtual environment,
- install all the project's dependencies into the virtual environment with known-good versions, and
- install the project itself into the virtual environment.
  :::::

:::::{tab-item} pip
:sync: pip
The whole process is a lot more tedious and manual if you use `pip` directly.
Once you have Python installed, you can first create a virtual environment with:
::::{tab-set}
:::{tab-item} macOS and Linux

```console
$ python3 -m venv .venv
$ source .venv/bin/activate
```

:::
:::{tab-item} Windows

```console
$ python3 -m venv .venv
$ .venv\Scripts\activate.bat
```

:::
::::
Then, you can install the project via:

```console
(.venv) $ pip install -ve.
```

:::::
::::::

We also recommend using [nox](https://nox.thea.codes/en/stable/).
Nox is a Python automation tool that allows you to define tasks in a `noxfile.py` and then run them with a single command.

::::{tab-set}
:::{tab-item} via `uv`
:sync: uv
The easiest way to install nox is via [uv](https://docs.astral.sh/uv/).

```console
$ uv tool install nox
```

:::
:::{tab-item} via `brew`
:sync: brew
If you use macOS, then nox is in Homebrew, and you can use

```console
$ brew install nox
```

:::
:::{tab-item} via `pipx`
:sync: pipx
If you prefer to use [pipx](https://pypa.github.io/pipx/), you can install nox with

```console
$ pipx install nox
```

:::
:::{tab-item} via `pip`
:sync: pip
If you prefer to use regular `pip` (preferably in a virtual environment), you can install nox with

```console
$ pip install nox
```

:::
::::

We define four convenient nox sessions in the `noxfile.py`:

- `tests` to run the tests
- `minimums` to run the Python tests with the minimum dependencies
- `lint` to run the Python code formatting and linting
- `docs` to build the documentation

These are explained in more detail in the following sections.

### Running Tests

The code base is tested by unit tests using the [pytest](https://docs.pytest.org/en/latest/) framework.
The corresponding test files can be found in the {code}`test/python` directory.
A {code}`nox` session is provided to conveniently run the Python tests.

```console
$ nox -s tests
```

The above command will automatically build the project and run the tests on all supported Python versions.
For each Python version, it will create a virtual environment (in the {code}`.nox` directory) and install the project into it.
We take extra care to install the project without build isolation so that rebuilds are typically very fast.

If you only want to run the tests on a specific Python version, you can pass the desired Python version to the {code}`nox` command.

```console
$ nox -s tests-3.12
```

:::{note}
If you don't want to use {code}`nox`, you can also run the tests directly using {code}`pytest`.

```console
(.venv) $ pytest test/python
```

This requires that you have the project installed in the virtual environment and the test dependency group installed.
:::

We provide an additional nox session {code}`minimums` that makes use of `uv`'s `--resolution=lowest-direct` flag to
install the lowest possible versions of the direct dependencies.
This ensures that the project can still be built and the tests pass with the minimum required versions of the dependencies.

```console
$ nox -s minimums
```

### Code Formatting and Linting

The Python code is formatted and linted using a collection of [pre-commit hooks](https://pre-commit.com/).
This collection includes:

- [ruff](https://docs.astral.sh/ruff/) -- an extremely fast Python linter and formatter, written in Rust.
- [mypy](https://mypy-lang.org/) -- a static type checker for Python code

There are two ways of using these hooks:

- You can install the hooks manually by running

  ```console
  $ pre-commit install
  ```

  in the project root directory.
  This will install the hooks in the {code}`.git/hooks` directory of the repository.
  The hooks will then be executed automatically when committing changes.

- You can use the {code}`nox` session {code}`lint` to run the hooks manually.

  ```console
  $ nox -s lint
  ```

  :::{note}
  If you don't want to use {code}`nox`, you can also run the hooks directly using {code}`pre-commit`.

  ```console
  $ pre-commit run --all-files
  ```

  :::

### Documentation

The code base is documented using [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings).
Every public function, class, and module should have a docstring that explains what it does and how to use it.
Ruff will check for missing docstrings and will explicitly warn you if you forget to add one.

We heavily rely on [type hints](https://docs.python.org/3/library/typing.html) to document the expected types of function arguments and return values.
For the compiled parts of the code base, we provide type hints in the form of stub files in the {code}`src/mqt/qecc` directory.

The Python API documentation is integrated into the overall documentation that we host on ReadTheDocs using the
[sphinx-autoapi](https://sphinx-autoapi.readthedocs.io/en/latest/) extension for Sphinx.

## Working on the Documentation

The documentation is written in [MyST](https://myst-parser.readthedocs.io/en/latest/index.html) (a flavour of Markdown) and built using [Sphinx](https://www.sphinx-doc.org/en/master/).
The documentation source files can be found in the {code}`docs/` directory.

On top of the API documentation, we provide a set of tutorials and examples that demonstrate how to use the library.
These are written in Markdown using [myst-nb](https://myst-nb.readthedocs.io/en/latest/), which allows to execute Python code blocks in the documentation.
The code blocks are executed during the documentation build process, and the output is included in the documentation.
This allows us to provide up-to-date examples and tutorials that are guaranteed to work with the latest version of the library.

You can build the documentation using the {code}`nox` session {code}`docs`.

```console
$ nox -s docs
```

This will install all dependencies for building the documentation in an isolated environment, build the Python package, and then build the documentation.
Finally, it will host the documentation on a local web server for you to view.

:::{note}
If you don't want to use {code}`nox`, you can also build the documentation directly using {code}`sphinx-build`.

```console
(.venv) $ sphinx-build -b html docs/ docs/_build
```

The docs can then be found in the {code}`docs/_build` directory.
:::

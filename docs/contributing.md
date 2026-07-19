<!--- This file has been generated from an external template. Please do not modify it directly. -->
<!--- Changes should be contributed to https://github.com/munich-quantum-toolkit/templates. -->

# Contributing

Thank you for your interest in contributing to MQT QECC! This document outlines
the development guidelines and how to contribute.

We use GitHub to [host code](https://github.com/munich-quantum-toolkit/qecc), to
[track issues and feature requests][issues], as well as accept
[pull requests](https://github.com/munich-quantum-toolkit/qecc/pulls). See
<https://docs.github.com/en/get-started/quickstart> for a general introduction
to working with GitHub and contributing to projects.

## Types of Contributions

Pick the path that fits your time and interests:

- 🐛 Report bugs:

  Use the _🐛 Bug report_ template at
  <https://github.com/munich-quantum-toolkit/qecc/issues>. Include steps to
  reproduce, expected vs. actual behavior, environment, and a minimal example.

- 🛠️ Fix bugs:

  Browse [issues][issues], especially those labeled "bug", "help wanted", or
  "good first issue". Open a draft PR early to get feedback.

- 💡 Propose features:

  Use the _✨ Feature request_ template at
  <https://github.com/munich-quantum-toolkit/qecc/issues>. Describe the
  motivation, alternatives considered, and (optionally) a small API sketch.

- ✨ Implement features:

  Pick items labeled "feature" or "enhancement". Coordinate in the issue first
  if the change is substantial; start with a draft PR.

- 📝 Improve documentation:

  Add or refine docstrings, tutorials, and examples; fix typos; clarify
  explanations. Small documentation-only PRs are very welcome.

- ⚡️ Performance and reliability:

  Profile hot paths, add benchmarks, reduce allocations, deflake tests, and
  improve error messages.

- 📦 Packaging and tooling:

  Improve build configuration, type hints/stubs, CI workflows, and platform
  wheels. Incremental tooling fixes have a big impact.

- 🙌 Community support:

  Triage issues, reproduce reports, and answer questions in Discussions:
  <https://github.com/munich-quantum-toolkit/qecc/discussions>.

## Guidelines

Please adhere to the following guidelines to help the project grow sustainably.
Contributions that do not comply with these guidelines or violate our
{doc}`ai_usage` may be rejected without further review.

### Core Guidelines

- ["Commit early and push often"](https://www.worklytics.co/blog/commit-early-push-often).
- Write meaningful commit messages, preferably using
  [gitmoji](https://gitmoji.dev) for additional context.
- Focus on a single feature or bug at a time and only touch relevant files.
  Split multiple features into separate contributions.
- Add tests for new features to ensure they work as intended.
- Document new features.
- Add tests for bug fixes to demonstrate the fix.
- Document your code thoroughly and ensure it is readable.
- Keep your code clean by removing debug statements, leftover comments, and
  unrelated code.
- Check your code for style and linting errors before committing.
- Follow the project's coding standards and conventions.
- Be open to feedback and willing to make necessary changes based on code
  reviews.

### AI-assisted contributions

We acknowledge the utility of AI-based coding agents (e.g., Claude Code, OpenAI
Codex, or GitHub Copilot) in modern software development. However, their use
requires a high degree of responsibility and transparency to maintain code
quality and licensing compliance.

Please carefully read and follow our dedicated {doc}`ai_usage` before submitting
any AI-assisted contribution. In short:
**You are responsible for every line of code you submit**, and a
**human must always be in the loop**. Agents may perform coding and GitHub tasks
within an explicitly authorized scope, but you must review their work and remain
accountable for the result. Every agent-authored or agent-edited public text
body must begin with `🤖 *AI text below* 🤖`; issue and pull request titles are
exempt. AI-assisted commits must include an
`Assisted-by: [Model Name] via [Tool Name]` footer. AI assistance must not be
used for contributions to issues labeled "good first issue".

If you use an agent, it will automatically read the provided {code}`AGENTS.md`,
which contains context and instructions to help the agent work on MQT QECC. For
Claude Code, create a symlink with {code}`ln -s AGENTS.md CLAUDE.md` so Claude
picks up the same file.

### Pull Request Workflow

- Create PRs early. Work-in-progress PRs are welcome; mark them as drafts on
  GitHub.
- Use a clear title, reference related issues by number, and describe the
  changes. Follow the PR template; only omit the issue reference if not
  applicable.
- CI runs on all supported platforms and Python versions to build, test, format,
  and lint. All checks must pass before merging.
- When ready, convert the draft to a regular PR and request a review from a
  maintainer. If unsure, ask in PR comments. If you are a first-time
  contributor, mention a maintainer in a comment to request a review.
- If your PR gets a "Changes requested" review, address the feedback and push
  updates to the same branch. Do not close and reopen a new PR. Respond to
  comments to signal that you have addressed the feedback. Do not resolve review
  comments yourself; the reviewer will do so once satisfied.
- If the reviewer suggested changes with explicit code suggestions as part of
  the comments, apply these directly using the GitHub UI. This attributes the
  changes to the reviewer and automatically resolves the respective comments
  (this is an exception to the rule above). If there are multiple suggestions
  that you want to apply at once, you can batch them into a single commit. Go to
  the "Files changed" tab of the PR, and then click "Add suggestion to batch"
  for each suggestion you want to include. Once you are done selecting
  suggestions, click "Commit suggestions". Only apply suggestions manually if
  using the GitHub UI is not feasible.
- Re-request a review after pushing changes that address feedback.
- Do not squash commits locally; maintainers typically squash on merge. Avoid
  rebasing or force-pushing before reviews; you may rebase after addressing
  feedback if desired.

### Working with CodeRabbit

We often use [CodeRabbit](https://www.coderabbit.ai/) for the initial review of
PRs. We use this tool to ease the workload on our maintainers and to counteract
the trend of sloppy AI-assisted contributions.

Once your PR is ready, an initial review by CodeRabbit can be requested via
{code}`@coderabbitai full review`. Just post this command as a PR comment on
GitHub. Any subsequent reviews can be requested via
{code}`@coderabbitai review`.

To get the most out of it and help the project maintain its high ambitions for
code quality, please follow these practices:

- **Review the review**: Do not take CodeRabbit's suggestions as absolute truth.
  LLMs can be overly defensive and conservative, leading to overcomplicated
  code. Treat its comments as suggestions: consider them, but feel free to
  disagree and explain why.
- **Respond to comments**: Do not simply resolve CodeRabbit's comments without
  answering them. It learns from your replies and improves over time. If a
  suggestion does not apply, take a moment to explain why in a reply.
- **Avoid multiple AI review bots**: CodeRabbit performs significantly worse
  when other AI review bots (e.g., GitHub Copilot) are active on the same PR.
  For the best results, do not tag Copilot unless you have already iterated with
  CodeRabbit and want an extra pass.
- **Engage CodeRabbit in discussions**: When team members are discussing code in
  PR comments, CodeRabbit stays silent by default. Tag {code}`@coderabbitai` to
  engage it in the conversation and get its feedback on the specific points
  being discussed. In particular, when you tag another person in a comment,
  ensure to also tag CodeRabbit. Otherwise, you will just get an automatic "It
  seems like the humans are having a chat" response from CodeRabbit anyway,
  which does not add much value.
- **Let CodeRabbit resolve comments**: Wait until after the next push before
  considering resolving CodeRabbit's comments manually. CodeRabbit will
  automatically resolve comments that it thinks have been addressed by your
  changes. Sometimes, it gets stuck, at which point you may resolve it manually.

Note that having your PR reviewed by CodeRabbit does **not** count as an
AI-assisted contribution for the purpose of the disclosure requirement mentioned
above.

## Get Started 🎉

Ready to contribute? We value contributions from people with all levels of
experience. In particular, if this is your first PR, not everything has to be
perfect. We will guide you through the process.

## Installation

Check out our {ref}`installation guide for developers <development-setup>` for
instructions on how to set up your development environment.

## Working on the Package

The package lives in the {code}`src/mqt/qecc` directory.

We recommend using [{code}`nox`][nox] for development. {code}`nox` is a Python
automation tool that allows you to define tasks in a {code}`noxfile.py` file and
then run them with a single command. If you have not installed it yet, see our
{ref}`installation guide for developers <development-setup>`.

We define some convenient {code}`nox` sessions in our {code}`noxfile.py`:

- {code}`tests` to run the Python tests
- {code}`minimums` to run the Python tests with the minimum dependencies
- {code}`lint` to run the Python code formatting and linting
- {code}`docs` to build the documentation

These are explained in more detail in the following sections.

## Running the Tests

The Python code is tested by unit tests using the
[{code}`pytest`](https://docs.pytest.org/en/latest/) framework. The
corresponding test files can be found in the {code}`tests` directory. A
{code}`nox` session is provided to conveniently run the Python tests.

```console
nox -s tests
```

This command automatically builds the project and runs the tests on all
supported Python versions. For each Python version, it will create a virtual
environment (in the {code}`.nox` directory) and install the project into it. We
take extra care to install the project without build isolation so that rebuilds
are typically very fast.

If you only want to run the tests on a specific Python version, you can pass the
desired Python version to the {code}`nox` command.

```console
nox -s tests-3.14
```

:::{note}

If you do not want to use {code}`nox`, you can also run the tests directly using
{code}`pytest`. This requires that you have the project and its test
dependencies installed in your virtual environment (e.g., by running
{code}`uv sync`).

```console
pytest
```

:::

We provide an additional nox session {code}`minimums` that makes use of
{code}`uv`'s {code}`--resolution=lowest-direct` flag to install the lowest
possible versions of the direct dependencies. This ensures that the project can
still be built and the tests pass with the minimum required versions of the
dependencies.

```console
nox -s minimums
```

## Code Formatting and Linting

The Python code is formatted and linted using a collection of pre-commit hooks.
This collection includes

- [ruff][ruff], an extremely fast Python linter and formatter written in Rust,
  and
- [ty][ty], Astral's type checker for Python.

The hooks can be installed by running the following command in the root
directory:

```console
prek install
```

This will install the hooks in the {code}`.git/hooks` directory of the
repository. The hooks will be executed whenever you commit changes.

You can also run the {code}`nox` session {code}`lint` to run the hooks manually.

```console
nox -s lint
```

:::{note}

If you do not want to use {code}`nox`, you can also run the hooks manually by
using [{code}`prek`][prek].

```console
prek run --all-files
```

:::

## Documentation

The Python code is documented using
[Google-style docstrings](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings).
Every public function, class, and module should have a docstring that explains
what it does and how to use it. {code}`ruff` will check for missing docstrings
and will explicitly warn you if you forget to add one.

We heavily rely on [type hints](https://docs.python.org/3/library/typing.html)
to document the expected types of function arguments and return values.

The Python API documentation is integrated into the overall documentation that
we host on ReadTheDocs using the
[{code}`sphinx-autoapi`](https://sphinx-autoapi.readthedocs.io/en/latest/)
extension for Sphinx.

(working-on-documentation)=

## Working on the Documentation

The documentation is written in
[MyST](https://myst-parser.readthedocs.io/en/latest/index.html) (a flavor of
Markdown) and built using [Sphinx](https://www.sphinx-doc.org/en/master/). The
documentation source files can be found in the {code}`docs/` directory.

On top of the API documentation, we provide a set of tutorials and examples that
demonstrate how to use the library. These are written in Markdown using
[myst-nb](https://myst-nb.readthedocs.io/en/latest/), which allows executing
Python code blocks in the documentation. The code blocks are executed during the
documentation build process, and the output is included in the documentation.
This allows us to provide up-to-date examples and tutorials that are guaranteed
to work with the latest version of the library.

You can build the documentation using the {code}`nox` session {code}`docs`.

```console
nox -s docs
```

This will install all dependencies for building the documentation in an isolated
environment, build the Python package, and then build the documentation. It will
then host the documentation on a local web server for you to view.

:::{note}

If you do not want to use {code}`nox`, you can also build the documentation
directly using {code}`sphinx-build`. This requires that you have the project and
its documentation dependencies installed in your virtual environment (e.g., by
running {code}`uv sync`).

```console
sphinx-build -b html docs/ docs/_build
```

The docs can then be found in the {code}`docs/_build` directory.

:::

## Tips for Development

If something goes wrong, the CI pipeline will notify you. Here are some tips for
finding the cause of certain failures:

- If any of the {code}`CI / 🐍 Test` checks fail, this indicates build errors or
  test failures. Look through the respective logs on GitHub for any error or
  failure messages.

- If any of the {code}`codecov/\*` checks fail, this means that your changes are
  not appropriately covered by tests or that the overall project coverage
  decreased too much. Ensure that you include tests for all your changes in the
  PR.

- If the {code}`pre-commit.ci` check fails, some of the {code}`prek` checks
  failed and could not be fixed automatically by the
  [pre-commit.ci](https://pre-commit.ci) bot. The individual log messages
  frequently provide helpful suggestions on how to fix the warnings.

- If the {code}`docs/readthedocs.org:\*` check fails, the documentation could
  not be built properly. Inspect the corresponding log file for any errors.

## Releasing a New Version

Before releasing a new version, check the GitHub release draft generated by the
Release Drafter for unlabelled PRs. Unlabelled PRs would appear at the top of
the release draft below the main heading. Furthermore, check whether the version
number in the release draft is correct. The version number in the release draft
is dictated by the presence of certain labels on the PRs involved in a release.
By default, a patch release will be created. If any PR has the {code}`minor` or
{code}`major` label, a minor or major release will be created, respectively.

:::{note}

Sometimes, Dependabot or Renovate will tag a PR updating a dependency with a
{code}`minor` or {code}`major` label because the dependency update itself is a
minor or major release. This does not mean that the dependency update itself is
a breaking change for MQT QECC. If you are sure that the dependency update does
not introduce any breaking changes for MQT QECC, you can remove the
{code}`minor` or {code}`major` label from the PR. This will ensure that the
respective PR does not influence the type of an upcoming release.

:::

Once everything is in order, navigate to the
[Releases page](https://github.com/munich-quantum-toolkit/qecc/releases) on
GitHub, edit the release draft if necessary, and publish the release.

<!--- Links --->
[nox]: https://nox.thea.codes/en/stable/
[prek]: https://prek.j178.dev
[ruff]: https://docs.astral.sh/ruff/
[ty]: https://docs.astral.sh/ty/
[issues]: https://github.com/munich-quantum-toolkit/qecc/issues

# Contributing

Thank you for your interest in contributing to this project.
We value contributions from people with all levels of experience.
In particular if this is your first pull request not everything has to be perfect.
We will guide you through the process.

We use GitHub to [host code](https://github.com/munich-quantum-toolkit/qecc), to [track issues and feature requests](https://github.com/munich-quantum-toolkit/qecc/issues), as well as accept [pull requests](https://github.com/munich-quantum-toolkit/qecc/pulls).
See <https://docs.github.com/en/get-started/quickstart> for a general introduction to working with GitHub and contributing to projects.

## Types of Contributions

You can contribute in several ways:

- 🐛 Report Bugs
  : Report bugs at <https://github.com/munich-quantum-toolkit/qecc/issues> using the _🐛 Bug report_ issue template. Please make sure to fill out all relevant information in the respective issue form.

- 🐛 Fix Bugs
  : Look through the [GitHub Issues](https://github.com/munich-quantum-toolkit/qecc/issues) for bugs. Anything tagged with "bug" is open to whoever wants to try and fix it.

- ✨ Propose New Features
  : Propose new features at <https://github.com/munich-quantum-toolkit/qecc/issues> using the _✨ Feature request_ issue template. Please make sure to fill out all relevant information in the respective issue form.

- ✨ Implement New Features
  : Look through the [GitHub Issues](https://github.com/munich-quantum-toolkit/qecc/issues) for features. Anything tagged with "feature" or "enhancement" is open to whoever wants to implement it. We highly appreciate external contributions to the project.

- 📝 Write Documentation
  : MQT QECC could always use some more documentation, and we appreciate any help with that.

## Get Started 🎉

Ready to contribute? Check out the {doc}`Development Guide <DevelopmentGuide>` to set up MQT QECC for local development and learn about the style guidelines and conventions used throughout the project.

We value contributions from people with all levels of experience.
In particular, if this is your first PR, not everything has to be perfect.
We will guide you through the PR process.
Nevertheless, please try to follow the guidelines below as well as you can to help make the PR process quick and smooth.

## Core Guidelines

- ["Commit early and push often"](https://www.worklytics.co/blog/commit-early-push-often).
- Write meaningful commit messages, preferably using [gitmoji](https://gitmoji.dev) for additional context.
- Focus on a single feature or bug at a time and only touch relevant files. Split multiple features into separate contributions.
- Add tests for new features to ensure they work as intended.
- Document new features appropriately.
- Add tests for bug fixes to demonstrate that the bug has been resolved.
- Document your code thoroughly and ensure it is readable.
- Keep your code clean by removing debug statements, leftover comments, and unrelated code.
- Check your code for style and linting errors before committing.
- Follow the project's coding standards and conventions.
- Be open to feedback and willing to make necessary changes based on code reviews.

## Pull Request Workflow

- Create PRs early. It is ok to create work-in-progress PRs. You may mark these as draft PRs on GitHub.
- Describe your PR with a descriptive title, reference any related issues by including the issue number in the PR description, and add a comprehensive description of the changes. Follow the provided PR template and do not delete any sections, except for the issue reference if your PR is not related to an issue.
- Whenever a PR is created or updated, several workflows on all supported platforms and versions of Python are executed. These workflows ensure that the project still builds, all tests pass, the code is properly formatted, and no new linting errors are introduced. Your PR must pass all these continuous integration (CI) checks before it can be merged.
- Once your PR is ready, change it from a draft PR to a regular PR and request a review from one of the project maintainers. Only request a review once you are done with your changes and the PR is ready to be reviewed. If you are unsure whether your PR is ready, ask in the PR comments. If you are a first-time contributor, request a review from one of the maintainers by mentioning them in a comment on the PR.
- If your PR gets a "Changes requested" review, address the feedback and update your PR by pushing to the same branch. Do not close the PR and open a new one. Respond to review comments on the PR (e.g., with "done 👍" or "done in @<commit>") to let the reviewer know that you have addressed the feedback. Note that reviewers do not get a notification if you just react to the review comment with an emoji. Write a comment to notify the reviewer. Do not resolve the review comments yourself. The reviewer will mark the comments as resolved once they are satisfied with the changes.
- Be sure to re-request a review once you have made changes after a code review so that maintainers know that the requests have been addressed.
- No need to squash commits before merging; we usually squash them to keep the history clean. We only merge without squashing if the commit history is clean and meaningful. Avoid rebasing or force-pushing your PR branch before merging, as it complicates reviews. You can rebase or clean up commits after addressing all review comments if desired.

Here are some tips for finding the cause of certain failures:

- If any of the `CI / 🐍 Test` checks fail, this indicates build errors or test failures in the Python part of the code base. Look through the respective logs on GitHub for any error or failure messages.
- If any of the `codecov/\*` checks fail, this means that your changes are not appropriately covered by tests or that the overall project coverage decreased too much. Ensure that you include tests for all your changes in the PR.
- If the `pre-commit.ci` check fails, some of the `pre-commit` checks failed and could not be fixed automatically by the _pre-commit.ci_ bot. Such failures are most likely related to the Python part of the code base. The individual log messages frequently provide helpful suggestions on how to fix the warnings.
- If the `docs/readthedocs.org:\*` check fails, the documentation could not be built properly. Inspect the corresponding log file for any errors.

---

This document was inspired by and partially adapted from

- <https://matplotlib.org/stable/devel/coding_guide.html>
- <https://opensource.creativecommons.org/contributing-code/pr-guidelines/>
- <https://yeoman.io/contributing/pull-request.html>
- <https://github.com/scikit-build/scikit-build>

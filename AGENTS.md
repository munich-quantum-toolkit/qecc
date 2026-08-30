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

- MUST read and follow `docs/ai_usage.md`. A human must review and understand
  all AI-assisted work. AI assistance must not be used for contributions to
  issues labeled `good first issue`.
- MUST run `uvx nox -s lint` after every batch of changes. This runs the full
  `prek` hook set from `.pre-commit-config.yaml` (including `ruff`, `typos`,
  `ty`, formatting, and metadata checks). All hooks must pass before submitting.
- MUST add or update tests for every code change, even if not explicitly
  requested.
- MUST write tests that protect intended behavior or reproduce a concrete
  regression. NEVER test provisional implementation choices that are not part of
  the supported contract.
- MUST place tests in the repository's corresponding test tree, organized by the
  component that owns the behavior. NEVER place tests or test fixtures in
  production source or tool directories. Reserve tool- or CLI-level subprocess
  tests for contracts that cannot be exercised meaningfully through a lower-
  level public API. Normal test targets and dependencies belong in the test
  build; avoid promoting an otherwise optional production tool into the default
  build solely for subprocess testing.
- MUST follow existing code style by checking neighboring files for patterns.
- MUST write code comments, documentation, tests, changelog entries, and public
  text for the final design. NEVER preserve prompts, review chronology, former
  names, or abandoned approaches unless they remain necessary user-facing
  context.
- MUST apply
  [Orwell's six rules for writing](https://www.orwellfoundation.com/the-orwell-foundation/orwell/essays-and-other-works/politics-and-the-english-language/)
  to every category of prose, including reasoning, descriptions, commit
  messages, documentation, docstrings, comments, test text, diagnostics, and
  handoffs:

  1. Do not use a familiar metaphor, simile, or other figure of speech.
  2. Use a short word when it has the same meaning as a long word.
  3. Remove every word that does not add meaning.
  4. Use active voice when possible.
  5. Use everyday English instead of a foreign phrase, scientific word, or
     jargon term when this does not reduce precision.
  6. Break a rule before it makes the text unclear, incorrect, or needlessly
     difficult to read.

- MUST apply the relevant principles of
  [ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/): use
  short, direct sentences; give each sentence one main idea; use one term for
  one meaning; and use explicit nouns instead of vague pronouns. These are
  mandatory style rules, not a claim of formal ASD-STE100 compliance.
- MUST base terminology and phrasing on existing usage in the repository and on
  established precedents in the communities the repository draws from. Use the
  established term that most precisely matches the concept. If communities use
  different terms, explain the mapping once. NEVER invent synonyms for variety.
- MUST remove obsolete scaffolding and diagnostic suppressions before handoff.
  Keep a workaround or suppression only when it is still necessary, scope it as
  narrowly as possible, and document the technical reason.
- AI assistance MUST be disclosed in the PR description.
- Commit-level `Assisted-by: [Model Name] via [Tool Name]` trailers are
  recommended, not required. For example:
  `Assisted-by: Claude Sonnet 4.6 via GitHub Copilot`. Do not rewrite otherwise
  valid history solely to add one.
- NEVER modify files that start with "This file has been generated from an
  external template. Please do not modify it directly." These files are managed
  by
  [the MQT templates action](https://github.com/munich-quantum-toolkit/templates)
  and changes will be overwritten.
- PREFER running targeted tests over the full test suite during development.

### GitHub Issues and Pull Requests

- MAY create, submit, and edit pull requests; create and manage issues; and
  comment on issues or pull requests when the task explicitly authorizes that
  external action. A single authorization may cover a clearly scoped task;
  per-message approval is not required. Actions outside that scope require fresh
  authorization. The human remains responsible for reviewing all submitted work.
- MUST use the repository's pull request template when one is present.
- Every agent-authored or agent-edited public text body MUST begin with
  `🤖 *AI text below* 🤖` on its first line. This applies to issue and pull
  request descriptions, review bodies, inline review comments, issue comments,
  replies, and other submitted text bodies; titles are exempt.
- When editing human-authored public text, preserve its original content and add
  the disclosure at the beginning of the edited field.
- MUST keep external communication accurate, specific, and non-repetitive; do
  not post low-quality or unsolicited comments.
- When reviewing a contribution, MUST focus findings on substantive correctness,
  contracts, maintainability, tests, documentation, licensing, and validation.
  NEVER report a missing optional AI-attribution trailer as a review finding.

### Python

- MUST use Google-style docstrings
- PREFER running a single Python version over the full test suite during
  development.
- PREFER fixing reported warnings over suppressing them (e.g., with `# noqa`
  comments for ruff); only add ignore rules when necessary and document why.
- PREFER fixing typing issues reported by `ty` before adding suppression
  comments (`# ty: ignore[code]`); suppressions are sometimes necessary for
  incompletely typed libraries (e.g., Qiskit).

## Self-Review Checklist

- Did `uvx nox -s lint` pass without errors?
- Are all changes covered by at least one automated test?
- Were any agent-authored issue or pull request texts explicitly authorized and
  marked with the required visible disclosure?

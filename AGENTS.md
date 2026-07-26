# Deep Learning Study Repository

This repository accompanies hands-on study of *Dive into Deep Learning* (D2L).
Favor clear, educational changes over production-oriented abstraction.

## Repository layout

- `notebooks/`: chapter-based experiments and implementations.
- `notes/`: mathematical derivations and conceptual study notes.
- `src/d2l_raw/`: reference D2L PyTorch source; do not modify unless explicitly requested.
- `src/d2l/`: editable local D2L helper package adapted for this environment.
- `uv_env/`: the pinned uv-managed Python environment and lockfile.

## Environment and commands

- Run Python or Jupyter commands from `uv_env/` with `uv`.
- The environment installs `../src/d2l` as an editable dependency, so notebook imports should use `import d2l.d2l as d2l`.
- After dependency or environment changes, verify with:

  ```sh
  cd uv_env && uv run python ../notebooks/test_env.py
  ```

## Notebook and note conventions

- Keep each notebook focused on a D2L concept or chapter section.
- Use Markdown cells or notes to explain non-obvious tensor shapes, gradients, assumptions, and training choices.
- Preserve worked examples and intermediate outputs when they aid study; avoid committing downloaded datasets, model checkpoints, or other large generated artifacts.
- Prefer small, direct PyTorch implementations before introducing helper abstractions.

## Changes and verification

- Do not change `src/d2l_raw/` when the same adjustment belongs in the editable `src/d2l/` package.
- Keep the local package compatible with the versions pinned in `uv_env/uv.lock`.
- Run the narrowest relevant verification after code changes; do not download datasets or run long training jobs unless requested.

# Python package layout and editable installs

This repository keeps the custom D2L helper package in:

```text
src/d2l/                         # installable project root
├── pyproject.toml               # package metadata and build configuration
└── src/                         # source-layout container
    └── d2l/                     # importable Python package
        ├── __init__.py
        └── d2l.py               # module imported as d2l.d2l
```

There are three distinct names here:

| Layer | Value | Purpose |
| --- | --- | --- |
| Project directory | `src/d2l` | Where the installable project lives in this repository. |
| Distribution name | `d2l` | The package name known to an installer such as uv. |
| Import package | `d2l` | The name used in Python: `import d2l`. |

The repeated `src/d2l/src/d2l` path is intentional. The first `src/d2l` is a repository location; the inner `src` is a packaging convention; and the final `d2l` is the Python import package.

## Why the notebook import works

The environment project declares this dependency in `uv_env/pyproject.toml`:

```toml
[tool.uv.sources]
d2l = { path = "../src/d2l", editable = true }
```

This asks uv to install the project at `../src/d2l` as an **editable** dependency. Instead of copying the package source into the virtual environment, the installation points Python at the working source directory. Changes to `src/d2l/src/d2l/d2l.py` are available on the next Python import or kernel restart without reinstalling the package.

Consequently, this notebook statement:

```python
import d2l.d2l as d2l
```

means:

1. Import package `d2l` from `src/d2l/src/d2l/`.
2. Import submodule `d2l.py` as `d2l.d2l`.
3. Bind that module to the notebook's local name `d2l`.

`src/d2l/src/d2l/__init__.py` also contains `from .d2l import *`, so this is another valid (but slightly different) style:

```python
import d2l
```

The notebooks use `import d2l.d2l as d2l` to refer explicitly to the module.

## How uv_build recognizes the src layout

The local package's `pyproject.toml` specifies:

```toml
[project]
name = "d2l"

[build-system]
requires = ["uv_build>=0.9.26,<0.10.0"]
build-backend = "uv_build"
```

The `build-system` entry selects `uv_build`. Its conventional defaults are:

```text
module root: src/
module name: normalized project name
expected package: src/<module-name>/__init__.py
```

Since the project name is `d2l`, the default expected package is `src/d2l/__init__.py`, which matches this project. No extra configuration is needed.

Python itself does not discover a “src layout.” The build backend creates the installation (or editable-install) mapping. Later, Python resolves `import d2l` using that installed mapping.

## Why use a src layout?

If the package instead used a flat layout:

```text
src/d2l/                         # project root
├── pyproject.toml
└── d2l/                         # import package
    ├── __init__.py
    └── d2l.py
```

then running Python from `src/d2l/` can make `import d2l.d2l` work merely because the current directory is on Python's import search path. That is convenient, but it can hide an incorrect package configuration: local tests may pass even if the package was not installed or a built wheel omits files.

With the src layout, running Python from the project root does not normally find `src/d2l/` by accident. Imports require an installation, including the editable installation used here. This makes development and testing more representative of how another user receives the package.

Do not use `import src.d2l.d2l` as a workaround. The inner `src` is not part of the package's intended public name; that import depends on the current working directory and does not test the installed package path.

## Using a flat layout with uv_build

`uv_build` defaults to a `src/` module root. If the inner `src/` directory is removed, opt into the flat layout explicitly:

```toml
[tool.uv.build-backend]
module-root = ""
```

The package directory would then be `d2l/` directly under the installable project root. The Python import remains unchanged:

```python
import d2l.d2l as d2l
```

For this repository, retain the current src layout: it is the default expected by the configured build backend and gives a useful guard against accidental imports from the project directory.

## Related files

- `src/d2l/pyproject.toml`: declares the installable distribution and selects `uv_build`.
- `uv_env/pyproject.toml`: declares the editable local dependency.
- `uv_env/uv.lock`: records `d2l` with `source = { editable = "../src/d2l" }`.
- `src/d2l/src/d2l/__init__.py`: package initializer.
- `src/d2l/src/d2l/d2l.py`: custom D2L helper module.

## Further reading

- [uv build backend documentation](https://docs.astral.sh/uv/concepts/build-backend/)
- [uv project packaging documentation](https://docs.astral.sh/uv/concepts/projects/config/)

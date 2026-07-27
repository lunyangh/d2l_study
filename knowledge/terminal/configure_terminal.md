# Terminal configuration snapshots

This directory records the shell and VS Code terminal configuration used in this Lightning AI Studio, plus the original local macOS configuration used as the visual reference.

## Snapshots

| Directory | Purpose |
| --- | --- |
| `lightningai/` | Reconstructed raw Lightning shell baseline before user-installed terminal tools. |
| `lightningai_configured/` | Snapshot of the current live Lightning configuration after the setup below. |
| `local/` | Local macOS reference configuration: Prezto, Spaceship, and Atom One Dark-oriented color choices. |

`lightningai_configured/.zshrc`, `.lightningrc`, and `settings.json` are direct copies of the live files at the time this document was written. The active files are outside this repository:

- `/teamspace/studios/this_studio/.zshrc`
- `/settings/.lightningrc`
- `/teamspace/studios/this_studio/.vscode-server/data/Machine/settings.json`

## Raw Lightning baseline

The reconstructed `lightningai/.zshrc` represents the relevant initial state:

- Oh My Zsh was installed and used `ZSH_THEME="robbyrussell"`.
- The Oh My Zsh plugin list was empty: `plugins=()`.
- The file sourced the Lightning-managed `/settings/.lightningrc` block.
- No user-installed zoxide, fzf, Spaceship, autosuggestion, or syntax-highlighting configuration was present.

The saved raw Lightning-managed file is named `lightningai/.lightingrc` for historical compatibility with the original snapshot. Its live source is `/settings/.lightningrc` (note the extra `n` in the filename).

## Current setup and installation order

The current Studio deliberately keeps Oh My Zsh as the only shell framework. It does not install Prezto.

1. **zsh-autosuggestions**
   - Cloned the official plugin into `$ZSH_CUSTOM/plugins/zsh-autosuggestions`.
   - Enabled it through `plugins=(zsh-autosuggestions)`.

2. **zoxide**
   - Ran the official zoxide installer.
   - It installed the binary at `$HOME/.local/bin/zoxide`.
   - Added `$HOME/.local/bin` to `PATH` before evaluating `zoxide init zsh`.

3. **fzf**
   - Lightning's optional Linuxbrew configuration was not available on this host, so Homebrew was not used.
   - Cloned the official fzf repository to `$HOME/.fzf` and ran `~/.fzf/install --bin` to install only its binary.
   - Added `$HOME/.fzf/bin` to `PATH` and loaded the current zsh integration with `source <(fzf --zsh)`.
   - fzf is optional for ordinary `z` navigation, but enables zoxide's interactive `zi` picker.

4. **Spaceship prompt**
   - Cloned Spaceship into `$ZSH_CUSTOM/themes/spaceship-prompt`.
   - Added the `spaceship.zsh-theme` symlink in the custom themes directory and changed `ZSH_THEME` to `spaceship`.
   - Started from the relevant `SPACESHIP_*` choices in `local/.zshrc`, then simplified the current prompt to an active virtual environment, yellow abbreviated directory, and `>>` prompt character. The username, host, battery, and Conda sections remain hidden; Git status stays on the right.
   - The virtual-environment section appears only when `$VIRTUAL_ENV` is set, with the form `venv:<name>`.
   - The terminal palette is supplied by VS Code's shared `Atom One Dark` theme, selected in `settings.json`.

5. **Autosuggestion and command colors**
   - Set `ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE="fg=gray,underline"` to make history suggestions visibly distinct.
   - Cloned `zsh-syntax-highlighting` into `$ZSH_CUSTOM/plugins/zsh-syntax-highlighting`.
   - Loaded it after zoxide and fzf, with Atom One Dark-compatible styles: yellow commands, cyan builtins/functions, red aliases, magenta paths/globs, bold green comments, white history expansion, and white-on-red `rm -rf` warnings.

## Ordering requirements and host caveats

- Put Spaceship and autosuggestion variables **before** `source $ZSH/oh-my-zsh.sh`; Oh My Zsh reads the theme and plugin configuration while it loads.
- Put the zoxide PATH entry before `eval "$(zoxide init zsh)"`; the official installer places zoxide in `$HOME/.local/bin`.
- Load `source <(fzf --zsh)` before `zsh-syntax-highlighting`.
- Source `zsh-syntax-highlighting` after prompt, fzf, zoxide, and other line-editor integrations. Loading it earlier can prevent it from highlighting commands reliably.
- Do not edit the Lightning-managed block in `.zshrc`, and do not edit `/settings/.lightningrc`. Lightning uses this code to attach VS Code terminals to persistent backend sessions.
- A regular VS Code integrated terminal normally enters the Lightning persistent session. The special `VSCODE_RESOLVING_ENVIRONMENT` shell does not; treat that variable as internal and do not set it manually.
- Open a new terminal after configuration changes. For a controlled shell-load check, use `DISABLE_SHELL_PERSISTENCE=1 zsh -ic '<check>'` so the test does not attach to a persistent session.
- Do not use the macOS `LSCOLORS` setting from `local/.zshrc` on Linux. Linux uses `LS_COLORS`; Lightning already supplies one.
- Confirm that the target host has Oh My Zsh, `git`, `curl`, and zsh before copying this setup. If the host has a different home directory, use `$HOME` and `$ZSH_CUSTOM` rather than hard-coded paths.
- Avoid installer modes that rewrite `.zshrc` automatically. The fzf binary-only install plus explicit configuration keeps the initialization order predictable.

## Quick validation on a new host

After recreating the setup, open a new VS Code terminal and confirm:

```zsh
print -r -- "$ZSH_THEME"
type z
type zi
typeset -f fzf-file-widget
typeset -f _zsh_autosuggest_start
typeset -f _zsh_highlight
```

Expected results are `spaceship`, shell functions for `z` and `zi`, and defined functions for fzf, autosuggestions, and syntax highlighting.

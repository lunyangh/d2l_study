---
name: lightning-studio-restore
description: Restore and verify the D2L Study repository after cloning it into a new Lightning AI Studio or after switching the same Studio to a different machine size, CPU/GPU type, or cloud hardware. Use when terminal tools, Oh My Zsh custom themes/plugins, the uv environment, or current machine resources need auditing or recovery.
---

# Lightning Studio Restore

Restore this repository using the committed terminal snapshots in `knowledge/terminal/lightningai_configured/` and the scripts bundled with this skill.

## Choose the scenario

- Use `fresh-clone` after cloning this repository into a new Lightning Studio. Sync the uv environment and, only with user approval, replace baseline shell and VS Code settings from the snapshots.
- Use `machine-switch` after moving the same Studio to any different machine. Preserve existing settings; audit first and restore only missing terminal assets. Report CPU, memory, and GPU availability without assuming that a GPU exists.

Do not treat a Studio transfer to another cloud provider as a machine switch. Inspect the destination first; it may require the fresh-clone path or a transfer-specific recovery.

## Restore the terminal

Run an audit first from the repository root:

```sh
.codex/skills/lightning-studio-restore/scripts/restore_terminal.sh \
  --scenario machine-switch
```

For a machine switch, restore only missing components after the audit:

```sh
.codex/skills/lightning-studio-restore/scripts/restore_terminal.sh \
  --scenario machine-switch --apply
```

For a fresh clone, inspect the existing `$HOME/.zshrc` and VS Code settings before replacing them. Explain that replacement creates timestamped backups, then run only after the user approves replacing both files:

```sh
.codex/skills/lightning-studio-restore/scripts/restore_terminal.sh \
  --scenario fresh-clone --apply --replace-config
```

The script restores these user-home assets when missing:

- Oh My Zsh custom `zsh-autosuggestions`, `zsh-syntax-highlighting`, and Spaceship theme/link.
- zoxide at `$HOME/.local/bin/zoxide`.
- fzf at `$HOME/.fzf/bin/fzf`.

Open a new terminal after restoration. Do not modify the Lightning-managed block in `.zshrc` or `/settings/.lightningrc`.

## Verify the uv environment and current machine

For a fresh clone, synchronize dependencies before verification:

```sh
.codex/skills/lightning-studio-restore/scripts/verify_uv.sh --sync --machine-check
```

For any machine switch, verify without synchronizing first:

```sh
.codex/skills/lightning-studio-restore/scripts/verify_uv.sh --machine-check
```

The verification uses `uv_env/` and `notebooks/test_env.py`. It does not download datasets or run training. With `--machine-check`, report logical CPU count, total and available memory, CUDA availability, and every detected GPU's name and memory. Treat `cuda_available=False` as a valid CPU-machine result, not an error.

## Handle failures

- If the terminal audit reports a missing Oh My Zsh framework, stop and ask before installing/replacing the framework; the script intentionally restores only its custom components.
- If `uv` or the UV check fails, report the failure before modifying the Python environment. For a fresh clone, propose `--sync`; for a machine switch, investigate persistent-environment loss first.
- If CUDA is unavailable, report the machine as CPU-only unless the user explicitly expected a GPU. Do not reinstall PyTorch automatically.
- Do not overwrite live configuration snapshots automatically. Update `knowledge/terminal/lightningai_configured/` only when the user intentionally changes the current configuration and wants the repository record refreshed.

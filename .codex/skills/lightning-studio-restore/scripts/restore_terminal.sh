#!/usr/bin/env bash
# Restore this repository's Lightning terminal dependencies.
set -euo pipefail

scenario=""
apply=false
replace_config=false

usage() {
  cat <<'USAGE'
Usage: restore_terminal.sh --scenario fresh-clone|machine-switch [--apply] [--replace-config]

Without --apply, report missing terminal components only.
--apply restores missing Oh My Zsh custom components, zoxide, and fzf.
--replace-config copies the saved .zshrc and VS Code settings after making backups.
Use --replace-config only for a fresh Studio whose configuration you intend to replace.
USAGE
}

while (($#)); do
  case "$1" in
    --scenario) scenario="${2:-}"; shift 2 ;;
    --apply) apply=true; shift ;;
    --replace-config) replace_config=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$scenario" in fresh-clone|machine-switch) ;; *) usage >&2; exit 2 ;; esac
if "$replace_config" && [[ "$scenario" != fresh-clone ]]; then
  printf '%s\n' '--replace-config is only allowed with --scenario fresh-clone.' >&2
  exit 2
fi

skill_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=$(cd -- "$skill_dir/../../.." && pwd)
snapshot="$repo_root/knowledge/terminal/lightningai_configured"
[[ -f "$snapshot/.zshrc" && -f "$snapshot/settings.json" ]] || {
  printf 'Missing terminal snapshots in %s\n' "$snapshot" >&2; exit 1;
}

home_dir=${HOME:?HOME must be set}
ohm_dir="${ZSH:-$home_dir/.oh-my-zsh}"
custom_dir="$ohm_dir/custom"
settings_dir="$home_dir/.vscode-server/data/Machine"

need=()
check_path() {
  local label=$1 path=$2
  if [[ -e "$path" || -L "$path" ]]; then
    printf 'present  %-28s %s\n' "$label" "$path"
  else
    printf 'missing  %-28s %s\n' "$label" "$path"
    need+=("$label")
  fi
}

printf 'Scenario: %s\n' "$scenario"
check_path 'Oh My Zsh' "$ohm_dir/oh-my-zsh.sh"
check_path 'autosuggestions' "$custom_dir/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh"
check_path 'syntax highlighting' "$custom_dir/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh"
check_path 'Spaceship source' "$custom_dir/themes/spaceship-prompt/spaceship.zsh"
check_path 'Spaceship link' "$custom_dir/themes/spaceship.zsh-theme"
check_path 'zoxide' "$home_dir/.local/bin/zoxide"
check_path 'fzf' "$home_dir/.fzf/bin/fzf"

if ! "$apply"; then
  ((${#need[@]} == 0)) || exit 1
  exit 0
fi

command -v git >/dev/null || { printf 'git is required.\n' >&2; exit 1; }

clone_if_missing() {
  local repository=$1 destination=$2
  [[ -e "$destination" ]] && return
  mkdir -p "$(dirname "$destination")"
  git clone --depth 1 "$repository" "$destination"
}

if [[ ! -f "$ohm_dir/oh-my-zsh.sh" ]]; then
  printf '%s\n' 'Oh My Zsh itself is missing. Install it first, then rerun this script.' >&2
  printf '%s\n' 'The script intentionally does not replace the shell framework automatically.' >&2
  exit 1
fi

clone_if_missing https://github.com/zsh-users/zsh-autosuggestions.git "$custom_dir/plugins/zsh-autosuggestions"
clone_if_missing https://github.com/zsh-users/zsh-syntax-highlighting.git "$custom_dir/plugins/zsh-syntax-highlighting"
clone_if_missing https://github.com/spaceship-prompt/spaceship-prompt.git "$custom_dir/themes/spaceship-prompt"
if [[ ! -e "$custom_dir/themes/spaceship.zsh-theme" && ! -L "$custom_dir/themes/spaceship.zsh-theme" ]]; then
  ln -s ../themes/spaceship-prompt/spaceship.zsh-theme "$custom_dir/themes/spaceship.zsh-theme"
fi

if [[ ! -x "$home_dir/.local/bin/zoxide" ]]; then
  command -v curl >/dev/null || { printf 'curl is required to install zoxide.\n' >&2; exit 1; }
  curl -sSfL https://raw.githubusercontent.com/ajeetdsouza/zoxide/main/install.sh | sh
fi

if [[ ! -x "$home_dir/.fzf/bin/fzf" ]]; then
  if [[ ! -d "$home_dir/.fzf/.git" ]]; then
    git clone --depth 1 https://github.com/junegunn/fzf.git "$home_dir/.fzf"
  fi
  "$home_dir/.fzf/install" --bin
fi

if "$replace_config"; then
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  for config in "$home_dir/.zshrc" "$settings_dir/settings.json"; do
    [[ -e "$config" ]] && cp "$config" "$config.bak-$stamp"
  done
  mkdir -p "$settings_dir"
  cp "$snapshot/.zshrc" "$home_dir/.zshrc"
  cp "$snapshot/settings.json" "$settings_dir/settings.json"
  printf 'Copied configured shell and VS Code snapshots; backups use .bak-%s.\n' "$stamp"
fi

printf '%s\n' 'Restoration complete. Open a new terminal, then run verify_uv.sh for the Python environment.'

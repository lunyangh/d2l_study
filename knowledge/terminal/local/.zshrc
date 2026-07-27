# change $PATH
export PATH="/usr/local/opt/ruby/bin:$PATH"
export PATH="$HOME/.gem/ruby/2.7.0/bin:$PATH"
# user installed packages
export PATH="$HOME/.local/bin:$PATH"

# set groovy
export GROOVY_HOME=/usr/local/opt/groovy/libexe
# set default editor to vim
export EDITOR=/usr/local/bin/vim
export VISUAL=/usr/local/bin/vim

# set google cloud dsk python interpreter
export CLOUDSDK_PYTHON=/Users/lunyanghuang/anaconda3/envs/benchmark_py311/bin/python3

 #Source Prezto.
if [[ -s "${ZDOTDIR:-$HOME}/.zprezto/init.zsh" ]]; then
  source "${ZDOTDIR:-$HOME}/.zprezto/init.zsh"
fi

# Customize to your needs...
alias vi=/usr/local/bin/vim
alias vim=/usr/local/bin/vim
alias gcc='gcc-11'
alias g++='g++-11'

# ****** jupyter notebook related *********
alias jb='jupyter lab --notebook-dir=.'

# setup ls colors
export LSCOLORS=exfxfeaeBxxehehbadacea
# export LSCOLORS=gxfxbEaEBxxEhEhBaDaCaD


t5='/Volumes/Ext_Data'


# ****** tmux related alias *********
# create a new session with name
alias tn='tmux new-session -s'
# attach to a session with name
alias ta='tmux attach -t'
# lists all ongoing sessions
alias tl='tmux list-sessions'
# kill session
alias tk='tmux kill-session -t'

# clear history, used within tmux panel
alias tcls='tmux clear-history'


# helper functions to run docker compose
alias dcrun='docker-compose run --service-ports --rm'


# set keybinding to vim
# bindkey -v
# enable emacs keybinding
bindkey -e

# open editor to edit long command
autoload -z edit-command-line
zle -N edit-command-line
bindkey "^X^Z" edit-command-line

# Customize spaceship prompt
# ORDER
SPACESHIP_PROMPT_ORDER=(
    conda
    battery
    user
    dir
    host
    vi_mode
    char
)
SPACESHIP_RPROMPT_ORDER=(
    git
)
# GENERAL SETUP
SPACESHIP_PROMPT_ADD_NEWLINE=false
SPACESHIP_PROMPT_SEPARATE_LINE=true
SPACESHIP_PROMPT_FIRST_PREFIX_SHOW=false
#SPACESHIP_PROMPT_DEFAULT_PREFIX="@"
#SPACESHIP_PROMPT_DEFAULT_SUFFIX=
# CHAR
SPACESHIP_CHAR_SYMBOL=">>"
SPACESHIP_CHAR_PREFIX=""
SPACESHIP_CHAR_SUFFIX=" "
# USER
# SPACESHIP_USER_PREFIX="" # remove `with` before username
# SPACESHIP_USER_SUFFIX="" # remove space before host
# TIME
SPACESHIP_TIME_SHOW=false
# HOST
# Result will look like this:
#   username@:hostname)
SPACESHIP_USER_SHOW=true
SPACESHIP_HOST_PREFIX="@:"
SPACESHIP_HOST_SUFFIX=" "

# DIR
SPACESHIP_DIR_PREFIX=""
SPACESHIP_DIR_SUFFIX=" "
SPACESHIP_DIR_TRUNC=1 # show only last directory
SPACESHIP_DIR_TRUNC_REPO=false # no special treatment of git repository
SPACESHIP_DIR_COLOR='yellow'
# GIT
# Disable git symbol
#SPACESHIP_GIT_SYMBOL="" # disable git prefix
#SPACESHIP_GIT_BRANCH_PREFIX="" # disable branch prefix too
# Wrap git in `git:...)`
SPACESHIP_GIT_PREFIX='git:'
SPACESHIP_GIT_SUFFIX=" "
#SPACESHIP_GIT_BRANCH_SUFFIX="" # remove space after branch name
## Unwrap git status from `[...]`
#SPACESHIP_GIT_STATUS_PREFIX=""
#SPACESHIP_GIT_STATUS_SUFFIX=""

# NODE
SPACESHIP_NODE_PREFIX="node:"
SPACESHIP_NODE_SUFFIX=" "
SPACESHIP_NODE_SYMBOL=""

# RUBY
SPACESHIP_RUBY_PREFIX="ruby:"
SPACESHIP_RUBY_SUFFIX=" "
SPACESHIP_RUBY_SYMBOL=""

# XCODE
SPACESHIP_XCODE_PREFIX="xcode:"
SPACESHIP_XCODE_SUFFIX=" "
SPACESHIP_XCODE_SYMBOL=""

# SWIFT
SPACESHIP_SWIFT_PREFIX="swift:"
SPACESHIP_SWIFT_SUFFIX=" "
SPACESHIP_SWIFT_SYMBOL=""

# GOLANG
SPACESHIP_GOLANG_PREFIX="go:"
SPACESHIP_GOLANG_SUFFIX=" "
SPACESHIP_GOLANG_SYMBOL=""

# DOCKER
SPACESHIP_DOCKER_PREFIX="docker:"
SPACESHIP_DOCKER_SUFFIX=" "
SPACESHIP_DOCKER_SYMBOL=""

# CONDA
SPACESHIP_CONDA_PREFIX=""
SPACESHIP_CONDA_COLOR='blue'
# VENV
SPACESHIP_VENV_PREFIX="venv:"
SPACESHIP_VENV_SUFFIX=" "

# PYENV
SPACESHIP_PYENV_PREFIX="python:"
SPACESHIP_PYENV_SUFFIX=" "
SPACESHIP_PYENV_SYMBOL=""

# Vi mode eabled
SPACESHIP_VI_MODE_SHOW=false
SPACESHIP_VI_MODE_INSERT='[I]'
SPACESHIP_VI_MODE_NORMAL='[N]'
SPACESHIP_VI_MODE_SUFFIX=""
SPACESHIP_VI_MODE_COLOR='magenta'
# eval spaceship_vi_mode_enable
# Source zsh autocomplete
# source '$HOME/Dropbox (Personal)/Projects/Terminal/custom_package/zsh/zsh-autocomplete/zsh-autocomplete.plugin.zsh'

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/lunyanghuang/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/lunyanghuang/anaconda3/etc/profile.d/conda.sh" ]; then
        echo "reach first option"
        . "/Users/lunyanghuang/anaconda3/etc/profile.d/conda.sh"
    else
        echo "reach export conda path"
        export PATH="/Users/lunyanghuang/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# see last answer by https://stackoverflow.com/questions/57660263/tmux-recognised-conda-env-but-still-use-the-default-pytho://stackoverflow.com/questions/57660263/tmux-recognised-conda-env-but-still-use-the-default-python
# add below lines at tail of your ~/.zshrc file, this for handling tmux
function conda_deactivate_all() {
        while [ -n "$CONDA_PREFIX" ]; do
                    conda deactivate;
                        done
                    }
[[ -z $TMUX ]] || conda_deactivate_all; conda activate base

# export PS1='%n-%1~:'

# eval "$(starship init zsh)"


# why would you type 'cd dir' if you could just type 'dir'?
# setopt AUTO_CD
unsetopt AUTO_CD

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh




source /Users/lunyanghuang/htcondor/condor/condor.sh


# init zoxide
eval "$(zoxide init zsh)"












# The next line updates PATH for the Google Cloud SDK.
if [ -f '/Users/lunyanghuang/Dropbox (Personal)/project_v2/google_cloud/cloud_cli/google-cloud-sdk/path.zsh.inc' ]; then . '/Users/lunyanghuang/Dropbox (Personal)/project_v2/google_cloud/cloud_cli/google-cloud-sdk/path.zsh.inc'; fi

# The next line enables shell command completion for gcloud.
if [ -f '/Users/lunyanghuang/Dropbox (Personal)/project_v2/google_cloud/cloud_cli/google-cloud-sdk/completion.zsh.inc' ]; then . '/Users/lunyanghuang/Dropbox (Personal)/project_v2/google_cloud/cloud_cli/google-cloud-sdk/completion.zsh.inc'; fi


# Added by Antigravity CLI installer
export PATH="/Users/lunyanghuang/.local/bin:$PATH"

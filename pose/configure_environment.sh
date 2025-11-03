#!/bin/bash
# This script sets up the shell environment for micromamba.

# >>> mamba initialize >>>
export MAMBA_EXE='/CT/eeg-3d-face/work/bin/micromamba'
export MAMBA_ROOT_PREFIX='/CT/eeg-3d-face/work/micromamba'
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE" # Fallback on help from micromamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<
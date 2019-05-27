#!/usr/bin/env bash

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$PYTHONPATH:$DIR_TMP/src
export PYTHONPATH=$PYTHONPATH:$DIR_TMP/utest
# export PYTHONPATH=$PYTHONPATH:$DIR_TMP/_90_dependencies
export PYTHONPATH=$PYTHONPATH:$DIR_TMP/90_dependencies/01_drqa
export PYTHONPATH=$PYTHONPATH:$DIR_TMP/90_dependencies/_02_core_nlp

echo PYTHONPATH=$PYTHONPATH
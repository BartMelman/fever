#!/usr/bin/env bash

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$PYTHONPATH:$DIR_TMP

export PYTHONPATH=$PYTHONPATH:$DIR_TMP/_10_scripts/_10_document_retrieval

echo PYTHONPATH=$PYTHONPATH
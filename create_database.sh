#!/bin/bash
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PYTHONPATH=src python3 src/scripts/build_db.py data/wiki-pages data/fever/fever.db
PYTHONPATH=src python3 src/scripts/build_tfidf.py data/fever/fever.db data/index/
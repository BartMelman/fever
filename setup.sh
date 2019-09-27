#!/usr/bin/env bash

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR='_01_data'
RAW_DATA_DIR='_01_raw_data'
WIKI_PAGES_DIR='_02_wikipedia_pages' 
DATABASE_DIR='_03_database'
# IR_DIR='03_ir'
CONFIG_DIR='_02_configs'
MODEL_DIR='_03_models'
RESULTS_DIR='_04_results'
SCRIPTS_DIR='_10_scripts'
SCRIPTS_DATABASE_DIR='_01_database'
TMP_DIR='_99_tmp'
DEPENDENCIES_DIR='_90_dependencies'
DRQA_DIR='_01_drqa'
CORE_NLP_DIR='_02_core_nlp'

mkdir -p $DATA_DIR/$WIKI_PAGES_DIR
mkdir -p $DATA_DIR/$DATABASE_DIR
# mkdir -p $DATA_DIR/$IR_DIR
mkdir -p $TMP_DIR
mkdir -p $RESULTS_DIR
mkdir -p $MODEL_DIR
mkdir -p $SCRIPTS_DIR
mkdir -p $DEPENDENCIES_DIR

download_if_not_exist() {
    if [ ! -f $2 ]; then
        wget $1 -O $2
    else
        echo "$2 already exists. skipping download..."
    fi
}

# === training, dev and test data === #
if [ ! -d $DATA_DIR/$RAW_DATA_DIR ]; then
	mkdir -p $DATA_DIR/$RAW_DATA_DIR
    download_if_not_exist https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl $DATA_DIR/$RAW_DATA_DIR/train.jsonl
    download_if_not_exist https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl $DATA_DIR/$RAW_DATA_DIR/dev.jsonl
    download_if_not_exist https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl $DATA_DIR/$RAW_DATA_DIR/test.jsonl
fi

# === wikipedia pages ==== #
if [ ! -d $DATA_DIR/$WIKI_PAGES_DIR/wiki-pages ]; then
    download_if_not_exist https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip $TMP_DIR/wiki-pages.zip
    mkdir -p $DATA_DIR/$WIKI_PAGES_DIR
    unzip $TMP_DIR/wiki-pages.zip -d $DATA_DIR/$WIKI_PAGES_DIR
    rm $TMP_DIR/wiki-pages.zip
fi

# === corenlp === #
# if [ ! -d $DEPENDENCIES_DIR/$CORE_NLP_DIR ]; then
#     # download_if_not_exist https://www.dropbox.com/s/74uc24un1eoqwch/dep_packages.zip?dl=0 $TMP_DIR/dep_packages.zip
#     download_if_not_exist http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip $TMP_DIR/dep_packages.zip
#     unzip $TMP_DIR/dep_packages.zip -d $DEPENDENCIES_DIR
#     mv $DEPENDENCIES_DIR/stanford-corenlp-full-2018-10-05 $DEPENDENCIES_DIR/$CORE_NLP_DIR
#     rm $TMP_DIR/dep_packages.zip
# fi

# === create data base === #
if [ ! -d $DATA_DIR/$DATA_BASE_DIR ]; then
    mkdir $DATA_DIR/$DATA_BASE_DIR
    PYTHONPATH=src python3 $SCRIPTS_DIR/$SCRIPTS_DATABASE_DIR/build_db.py $DATA_DIR/$WIKI_PAGES_DIR/wiki-pages $DATA_DIR/$DATABASE_DIR/wiki.db
fi

# === drqa === #
# if [ ! -d $DEPENDENCIES_DIR/$DRQA_DIR ]; then
#     git clone https://github.com/facebookresearch/DrQA.git $DEPENDENCIES_DIR/$DRQA_DIR
# fi

# # === create if idf data base === #
# if [ ! -d $DEPENDENCIES_DIR/$DATABASE_DIR/if_idf.db ]; then
#     python3 $SCRIPTS_DIR/$SCRIPTS_DATABASE_DIR/build_tfidf.py $DATA_DIR/$DATABASE_DIR/if_idf.db $DATA_DIR/$DATABASE_DIR/
# fi

# === spacy english tokenizer === #
python3 -m spacy download en

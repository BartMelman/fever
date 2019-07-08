import os
import json
from sqlitedict import SqliteDict
import shutil
from tqdm import tqdm
import spacy

from wiki_database import Text
from utils_db import dict_save_json, dict_load_json, load_jsonl, dict_save_json, write_jsonl
from vocabulary import VocabularySqlite
from vocabulary import count_n_grams
from tfidf_database import TFIDFDatabaseSqlite

import config

def add_score_to_results(score, method, results, i):
    results_key_name = 'results'
    if results_key_name not in results[i]:
        results[i][results_key_name] = {}

    results[i][results_key_name][method] = score

    return results

class Claim:
    def __init__(self, claim_dictionary):
        self.id = claim_dictionary['id']
        self.verifiable = claim_dictionary['verifiable']
        self.label = claim_dictionary['label']
        self.claim = claim_dictionary['claim']
        self.claim_without_dot = self.claim[:-1]
        self.evidence = claim_dictionary['evidence']
        # self.nlp = nlp

        if 'docs_selected' in claim_dictionary:
            self.docs_selected = claim_dictionary['docs_selected']
    
class ClaimDocTokenizer:
    def __init__(self, doc, delimiter_words):
        self.doc = doc
        self.delimiter_words = delimiter_words
    def get_tokenized_claim(self, method_tokenization):
        # claim_without_dot = self.claim[:-1]  # remove . at the end
        # doc = self.nlp(claim_without_dot)
        text = Text(self.doc)
        tokenized_claim = text.process(method_tokenization)
        return tokenized_claim
    def get_n_grams(self, method_tokenization, n_gram):
        return count_n_grams(self.get_tokenized_claim(method_tokenization), n_gram, 'str', self.delimiter_words)

def get_tag_word_from_wordtag(key, delimiter):
    splitted_key = key.split(delimiter)

    return splitted_key[0], splitted_key[1]

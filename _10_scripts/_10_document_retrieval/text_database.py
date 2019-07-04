import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import spacy

from _10_scripts._01_database.wiki_database import WikiDatabase

class TextDatabaseSqlite:
    """A sample Employee class"""
    def __init__(self, wiki_database):
        self.wiki_database = wiki_database
        
        self.nlp = spacy.load('en', disable=["parser", "ner"])

    def get_tokenized_text_from_id(self, id_nr, method_list = []):
        # method: 
        text = Text(self.wiki_database.get_text_from_id(id_nr), 'text', self.nlp)
        tokenized_text = text.process(method_list)
        return tokenized_text
    
    def get_tokenized_title_from_id(self, id_nr, method_list = []):
        # recover title from wiki_database and return the tokonized title
        title = Text(self.wiki_database.get_title_from_id(id_nr), 'title', self.nlp)
        tokenized_title = title.process(method_list)
        return tokenized_title
    

class TextDatabase:
    """A sample Employee class"""
    def __init__(self, path_wiki_database, table_name_wiki):
        self.table_name = table_name_wiki
        self.wiki_database = WikiDatabase(path_wiki_database, table_name_wiki)
        self.nr_rows = self.wiki_database.nr_rows
        
        self.word_list = []
        self.nr_words = None
        self.nlp = spacy.load('en', disable=["parser", "ner"])

    def get_tokenized_text_from_id(self, id_nr, method_list = []):
        # method: 
        text = Text(self.wiki_database.get_text_from_id(id_nr), 'text', self.nlp)
        tokenized_text = text.process(method_list)
        return tokenized_text
    
    def get_tokenized_title_from_id(self, id_nr, method_list = []):
        # recover title from wiki_database and return the tokonized title
        title = Text(self.wiki_database.get_title_from_id(id_nr), 'title', self.nlp)
        tokenized_title = title.process(method_list)
        return tokenized_title
    
# def get_tokenized_claim(self, claim, method_list = []):
#     text = Text(claim, 'text', self.nlp)
#     tokenized_text = text.process(method_list)
#     return tokenized_text
    
class Text:
    """A sample Employee class"""
    def __init__(self, text):
        # self.delimiter_title = '_'
        # self.delimiter_text = ' '
        self.list_accepted_tags = ['INTJ', 'NOUN', 'NUM', 'PART', 'PROPN', 'SYM', 'X']
        self.list_nouns = ['NOUN', 'PROPN']
        self.list_symb = ['SYM']
        self.list_num = ['NUM']
        self.list_prop_noun = ['PROPN']
        # text_type_options = ['title', 'text', 'lines', 'claim']

        self.text = text
        self.delimiter_position_tag = '\z'
        
        # if text_type not in text_type_options:
        #     raise ValueError('text_type not in text_type_options', text_type, text_type_options)
        # else:
        #     self.text_type = text_type
        
        # if text_type == 'title':
        #     self.text = self.text.replace(self.delimiter_title, self.delimiter_text)

    def process(self, method_list):
        """Dispatch method"""
        method_options = ['tokenize', 'tokenize_lemma', 'tokenize_lemma_list_accepted', 
        'tokenize_lemma_nouns', 'tokenize_lemma_prop_nouns', 'tokenize_lemma_number', 'tokenize_lemma_pos'] 

        for method in method_list:
            if method not in method_options:
                raise ValueError('method not in method_options', method, method_options)
            method = getattr(self, method, lambda: "Method Options")
            text = method()
            
        return text
    
    def tokenize(self):
        return [word.text for word in self.text]

    def tokenize_lemma(self):
        return [word.lemma_ for word in self.text]

    def tokenize_lemma_list_accepted(self):
        return [word.lemma_ for word in self.text if word.pos_ in self.list_accepted_tags]

    def tokenize_lemma_nouns(self):
        return [word.lemma_ for word in self.text if word.pos_ in self.list_nouns]

    def tokenize_lemma_prop_nouns(self):
        return [word.lemma_ for word in self.text if word.pos_ in self.list_prop_noun]

    def tokenize_lemma_number(self):
        return [word.lemma_ for word in self.text if word.pos_ in self.list_num]

    def tokenize_lemma_pos(self):
        return [word.pos_ + self.delimiter_position_tag + word.lemma_ for word in self.text]

    def tokenize_text_pos(self):
        return [word.pos_ + self.delimiter_position_tag + word.text for word in self.text]

    def tokenize_lower_pos(self):
        return [(word.pos_ + self.delimiter_position_tag + word.text).lower() for word in self.text]

#     def tokenize(self, text):
#         # definition: replace '-LRB-' by '-LRB- ' if it is a title.
#         if self.text_type == 'title':
#             tokenized_text = text.replace('-LRB-', '-LRB-%s'%(self.delimiter)).replace('-RRB-', '%s-RRB-'%(self.delimiter))
#         else:
#             tokenized_text = text
#         tokenized_text = tokenized_text.split(self.delimiter)
           
#         return tokenized_text
    
#     def remove_bracket_and_word_between(self, input_list):
#         left_bracket_symbol = '-LRB-'
#         right_bracket_symbol = '-RRB-'
    
#         left_ids = find_ids(input_list, left_bracket_symbol)
#         right_ids = find_ids(input_list, right_bracket_symbol)

#         if len(left_ids) != len(right_ids):
#             print('The number of left brackets and right brackets is not equal')
# #             raise ValueError('The number of left brackets and right brackets is not equal.', len(left_ids), len(right_ids), left_ids, right_ids)
#             word_list_without_brackets = input_list
#         else:
#             list_remove = [j for i in range(len(left_ids)) for j in range(left_ids[i], right_ids[i] + 1)]

#             word_list_without_brackets = [input_list[i] for i in range(len(input_list)) if i not in list_remove]

#         return word_list_without_brackets
    
#     def remove_space(self, word_list):
#         word_list_without_space = [word for word in word_list if word is not '']
#         return word_list_without_space
    
#     def make_lower_case(self, text):
#         output_word_list = [word.lower() for word in text]
#         return output_word_list
    
#     def lemmatization(self, word_list):
#         lemmatized_text = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in word_list]
#         return lemmatized_text
    
#     def lemmatization_get_nouns(self, word_list):
#         lemmatizer = WordNetLemmatizer()
#         list_selected_tags = ['CD', 'FW', 'NN', 'NNS','NNP','NNPS']
#         tagged_word_list = nltk.pos_tag(word_list) 
#         reduced_word_list = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for (word, tag) in tagged_word_list if tag in list_selected_tags]
#         return reduced_word_list
    
def find_ids(input_list, search_string):
    list_ids = [i for i, j in enumerate(input_list) if j == search_string]
    return list_ids

# def get_wordnet_pos(word):
#     # """Map POS tag to first character lemmatize() accepts"""
#     tag = nltk.pos_tag([word])[0][1][0].upper()
#     tag_dict = {"J": wordnet.ADJ,
#                 "N": wordnet.NOUN,
#                 "V": wordnet.VERB,
#                 "R": wordnet.ADV}

#     return tag_dict.get(tag, wordnet.NOUN)
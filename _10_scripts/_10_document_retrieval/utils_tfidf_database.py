class DictionaryBatchIDF():
    def __init__(self, delimiter):
        self.dictionary_tf_idf  = {}
        self.delimiter = delimiter
                
    def update(self, word, doc_nr, tf_idf):
        try:
            self.dictionary_tf_idf[word]['values'] = self.dictionary_tf_idf[word]['values'] + self.delimiter + str("%.3f"%(tf_idf))
            self.dictionary_tf_idf[word]['ids'] = self.dictionary_tf_idf[word]['ids'] + self.delimiter + str(doc_nr)
        except KeyError:
            self.dictionary_tf_idf[word] = {'values': str("%.3f"%(tf_idf)), 'ids' : str(doc_nr)}
   
    def reset(self):
        self.dictionary_tf_idf = {}
        self.dictionary_value_str = {}
        self.dictionary_id_str = {}
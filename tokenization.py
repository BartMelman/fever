import pickle
import os
import config
# pickle_off = open("Emp.pickle","rb")
# emp = pickle.load(pickle_off)
# print(emp)
import drqa_yixin.tokenizers
# from drqa_yixin.tokenizers import CoreNLPTokenizer
import sys

__root__ = os.getcwd()
__data_dir__ = '_01_data'
__wiki_dir__ = '_02_wikipedia_pages'
__dep_dir__ = '_90_dependencies'

sys.path.append(os.path.join(__root__, __dep_dir__, 'DrQA'))
print(sys.path)
path_wiki_pages = os.path.join(__root__, __data_dir__, __wiki_dir__,'wiki-pages')


def num_files_in_directory(path_directory):
    # description: find the number of files in directory
    # input: path to directory
    # output: number of files in directory

    number_of_files = len([item for item in os.listdir(path_directory) if os.path.isfile(os.path.join(path_directory, item))])

    return number_of_files

if __name__ == '__main__':
    nr_wikipedia_files = num_files_in_directory(path_wiki_pages)
    print(nr_wikipedia_files)

    # path_stanford_corenlp_full_2017_06_09 = os.path.join(__root__, __dep_dir__, 'stanford-corenlp-full-2017-06-09/*')
    # print(path_stanford_corenlp_full_2017_06_09)
    # drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    # tok = CoreNLPTokenizer(annotators=['pos', 'lemma'])


    # d_list = load_jsonl(in_file)
    # for item in tqdm(d_list):
    #     item['claim'] = ' '.join(easy_tokenize(item['claim'], tok))

    # save_jsonl(d_list, out_file)
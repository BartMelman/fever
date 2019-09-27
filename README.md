# FEVER
The code in this repository has three functions: 
- (1) Automatically download all the data from the FEVER Challenge, 
- (2) Create databases to access the wikipedia pages/claims
- (3) Create databases with the tokenized text/claim

The advantages of this approach are:
- (1) The text in the databases are all stored in a common utf format. The raw data from the wikipedia pages and claims have different utf fromats. In the database creation process, I convert it to a common utf format.
- (2) The Spacy package is used for tokenization and is better than tokenization with e.g. nltk, because it takes context into conisderation. This is costly and therefore the tokenized text is directly stored, so it can be retrieved quickly. 
- (3) The databases are created with end-to-end checks to esure that the databases are constructed fully. If an error occurs and the database is called again, then an error is automatically generated and the database cannot be accessed. 

# computer settings
- python3
- laptop

# setup (only once): 
description: this script sets up the folder structure and downloads the wikipedia and claim dataset 

    $ bash setup.sh

description: add all packages to virtual environment

    $ pip3 install -r requirements.txt

# startup
description: this script sets the paths to the different directories in the folder. 
This script needs to be called every time a command prompt is started.

    $ source set_paths.sh

# train databases
description: 

    $ cd _10_scripts/_10_document_retrieval
    $ python3 wiki_database.py
    $ python3 claim_database.py
    $ python3 wiki_database_n_grams.py
    $ python3 claim_database_n_grams.py

# tutorial
description: run the jupyter notebook tutorial.ipynb to train and investigate the databases

    $ jupyter notebook

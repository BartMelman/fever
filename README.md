# fever
description:

# computer settings
- python3

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

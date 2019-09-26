# fever
description:

# computer settings
- python3

# startup (only once): 
description: this script sets up the folder structure and downloads the wikipedia and claim dataset 
bash setup.sh

description: add all packages to virtual environment
pip install -r requirements.txt

# at start of command prompt
description: this script sets the paths such that the files in 


# Startup

mkdir -p 01_university/01_thesis

sudo apt-get install python3-venv
python3 -m venv 02_environment/01_fever
source 02_environment/01_fever/bin/activate

cd 01_university/01_thesis

# fever
description:

# computer settings
- python3

# setup (only once): 
description: this script sets up the folder structure and downloads the wikipedia and claim dataset 
  $ bash setup.sh

description: add all packages to virtual environment
  $ pip install -r requirements.txt

# startup
description: this script sets the paths to the different directories in the folder. 
This script needs to be called every time a command prompt is started.

  $ source set_paths.sh


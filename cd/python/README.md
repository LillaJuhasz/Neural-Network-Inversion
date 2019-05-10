* Description of how to load the source code of the implementation *


The source code of the implementation can be found on the attached media. As a requirement, Anaconda must be installed since it is the interpreter for running Python files, hence no Python IDE have to be downloaded. 
To load the source code, the environment must be set by the assistance of Anaconda's package manager system called conda. Since there are no integrated environment for Python, a virtual environment has to be established. To accomplish it, the following command has to be written in the conda prompt: 

>>> conda create -n venv python=3.6


Then the virtual environment venv must be activated.

>>> conda activate venv


For running the source code, the necessary packages need to be loaded, that are located in the requirements.txt file.

>>> conda install --file requirements.txt


Finally the virtual environment is ready to load the Python file.
In the folder python, there are two .py files. The inversion.py contains the implementation of the inversion and the main.py includes everything else. Hence the main.py has to be loaded by the virtual environment with the following command: 

>>> python main.py


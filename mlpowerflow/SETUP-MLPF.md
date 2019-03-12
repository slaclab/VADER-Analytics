## Setting up your environment
Follow the steps below to install all dependencies, libraries, and packages to run ML based Power Flow notebook.

The tutorial uses Anaconda distribution of Python 3.6

### Installing Anaconda
[Install the Anaconda distro](https://www.anaconda.com/distribution/)
- Make sure to select your appropriate OS
- Also, highly recommend installing the python `3.x` version - it'll allow you to create environments in python `2.x` if you want. But if you install Anaconda/python2.7, you won't be able to create `3.x` python envs.

Run the following commands to get setup:
```bash
# create the python 3.6 env via anaconda
conda create --name venv_mlpf python=3.6

# activate the env
conda activate venv_mlpf

# install the dependencies
conda install jupyter
conda install numpy
conda install pandas
conda install scikit-learn
conda install matplotlib
pip install pypower
pip install pandapower
# Finally, run the project
jupyter notebook
```

### Once the project is running
Your browser should automatically launch a page pointed at `localhost:8888` with Jupyter running. If that's not the case, please open your favorite browser and navigate to `localhost:8888`.
Then, simply select the `Sample_Script.ipynb`

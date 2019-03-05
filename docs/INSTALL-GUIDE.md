## Getting up and running
You have a couple of options to get going with this project. Choose the option you are most comfortable with.
You can use:
- Anaconda & python `2.7` (rec/easiest)
- Anaconda & python `3.6`
- Docker.

### Installing Anaconda
[Install the Anaconda distro](https://www.anaconda.com/distribution/)
- Make sure to select your appropriate OS
- Also, highly recommend installing the python `3.7` version - it'll allow you to create environments in python `2.x` if you want. But if you install Anaconda/python2.7, you won't be able to create `3.x` python envs.

### Using Anaconda & python `2.7`:
#### Caveat: _using anaconda to manage everything, easily, forces python 2.x due to a build conflict in the cvxpy lib. If you want to use python 3.x, please jump down to the next option._

Run the following command to create your conda environment
```bash
# create the env
conda env create -f environment.yml

# You should now have an environment named venv_vader_solar_disagg
# To check that this is the case, run
conda info --envs

# Activate the environment
conda activate venv_vader_solar_disagg

# Lastly, run the project
jupyter notebook
```

### Using Anaconda & python `3.6` (need to manually install dependencies to resolve conflicts):

Run the following commands to get setup:
```bash
# create the python 3.6 env via anaconda
conda create --name venv_vader_solar_disagg_py3 python=3.6

# activate the env
conda activate venv_vader_solar_disagg_py3

# install the dependencies
conda install jupyter
conda install numpy
conda install pandas
conda install scikit-learn
conda install matplotlib
conda install sqlalchemy
conda install seaborn
conda install psycopg2
pip install cvxpy==0.4.10

# Lastly, run the project
jupyter notebook
```

### Using Docker:
[Install Docker](https://docs.docker.com/install/)
```bash
# run the following commands
docker build --tag=vader_solar_disagg .
docker run -it -p 8888:8888 vader_solar_disagg
# docker will output a URL with a token that you should just
# copy and paste into your browser to launch the project
```

### Once the project is running
Your browser should automatically launch a page pointed at `localhost:8888` with Jupyter running. If that's not the case, please open your favorite browser (_I hope it's not IE_) and navigate to `localhost:8888`.
Then, simply select the `SolarDisagg_Individual_Home_Tutorial_Main.ipynb`

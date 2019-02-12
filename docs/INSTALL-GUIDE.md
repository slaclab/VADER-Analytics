## Getting up and running
You have a couple of options to get going with this project. Choose the option you are most comfortable with. You can use, Anaconda (rec/easiest), Virtualenv or Docker.

### Using Anaconda:
[Install the Anaconda distro](https://www.anaconda.com/distribution/)
- Make sure to select your appropriate OS
- Also, highly recommend installing the python 3.7 version - it'll allow you to create environments in python 2.x if you want. But if you install Anaconda/python2.7, you won't be able to create 3.x python envs.

#### Caveat: _using anaconda to manage everything, easily, forces python 2.x due to a build conflict in the cvxpy lib. If you want to use python 3.x, please jump down to the virtualenv option._

Run the following command to create your conda environment
```bash
conda env create -f environment.yml
```
You should now have an environment named `venv_vader_solar_disagg`.
To check that this is the case, run:
```bash
conda info --envs
```

Activate the environment:
```bash
conda activate venv_vader_solar_disagg
```

Run solar disagg:
```bash
jupyter notebook
```

Your browser should automatically launch a page pointed at `localhost:8888` with Jupyter running. If that's not the case, please open your favorite browser (_I hope it's not IE_) and navigate to `localhost:8888`.
Then, simply select the `SolarDisagg_Individual_Home_Tutorial_Main.ipynb`

### Using virtualenv

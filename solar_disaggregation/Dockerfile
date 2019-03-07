# get a miniconda python 3 base image
FROM continuumio/miniconda3

# update base libs
RUN apt-get update && apt-get install -y build-essential g++ libgl1-mesa-glx libx11-6  

# install project dependencies via conda...
RUN conda install jupyter numpy pandas scikit-learn matplotlib sqlalchemy seaborn psycopg2
# ... with the exception of cvxpy, here we need to use pip
RUN pip install cvxpy

# set the appropriate dir
WORKDIR /solar_disagg
COPY . /solar_disagg

# now go!
CMD jupyter notebook --allow-root --ip 0.0.0.0 --no-browser
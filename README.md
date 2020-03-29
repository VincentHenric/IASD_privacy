# Robust De-anonymisation of Large Sparse Datasets - IASD Privacy project

This repository contains the code, experiments results and report for the Privacy course project. 
The goal was to reproduce the results of the "Robust De-anonymisation of Large Sparse Datasets" paper by Arvind Narayanan and Vitaly Shmatikov (https://www.cs.utexas.edu/~shmat/shmat_oak08netflix.pdf) and to introduce some extensions. 

## Requirements

The project make use of several Python libraries, they can be installed using `pip install -r requirements.txt`:
- `numpy`,`scipy`: scientific computing
- `pandas`: data organisation, using dataframes
- `pyspark`: Python interface to Spark: distributed computing / local multi-processing and efficient memory management for data-intensive algorithms
- `matplotlib`,`seaborn`: results visualization 

## Data preparation

In order to reproduce the article results we use the original Netflix Prize Dataset that is available on Kaggle. 
- Download the dataset from the website (requires logging in): https://www.kaggle.com/netflix-inc/netflix-prize-data
- Place the `netflix-prize-data.zip` file in the `datasets/` directory
- Use `make netflix` to prepare the dataset. Basically it creates a single file `ratings.csv` that contains the 100M ratings.
- IMDB public data has also been analysed to see to what extent movies in IMDB could be linked to anonymous movie IDs in Netflix using other data such as ratings and popularity. This data can automatically be downloaded and extracted using `make imdb`.

## Project structure and files

- `config.py`: Spark settings.
- `privacy.py`: First implementation of the de-anonymisation algorithm using Pandas.
- `privacy_spark.py`: PySpark implementation: more heavyweight but it allows to experiment on the full dataset in a reasonable amount of time.
- `experiment.py`: Experiment runner. It performs de-anonymisation using various settings and store the results in the `experiments/` folder.
- `datasets/`: Dataset files location and extraction code.
- `experiments/`: Experiments results that can be used for further analysis. 
- `privacy_project.ipynb`: ...
- `privacy_project_spark.ipynb`: ...
- `result_analysis.ipynb`: Jupyter Notebook that reproduce some results of the article, and explore threshold selection.

## More details

(note: `privacy.py` and `privacy_spark.py` should provide the same features and only differ by their implementation)

### `privacy.py:Auxiliary`

Represents an auxiliary information generator. It's parameters are the error margins on rating and date.

### `privacy.py:Generate`

Generates auxiliary information according to the requested `Auxiliary` generators.

### `privacy.py:Scoreboard_RH`

Main algorithm for de-anonymisation as introduced by the article. This class provide the following features: matching score computation and de-anonymisation, either entropic or by best-guess.

### `privacy.py:Scoreboard_RH_without_movie` class

Our developped extension has been to experiment with the absence mapping between auxiliary movie IDs and real movie IDs. This makes the problem much harder and involves a less efficient algorithm.

### `experiment.py:Experiment` class

Wraps everything under a nice interface to build and save experiments.

## Reproducing results of the article

When the `experiment.py` script is run it launches a set of experiments that can then be analysed in `result_analysis.ipynb`. This code can be seen as an example on how everything works together.

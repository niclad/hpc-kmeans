# Hetergeneous Evaluation using K-Means Clustering

This repository contains all the code used to run k-means in several configurations.

## Configurations

Below are explanations of the configurations for each directory

### Serial

The [serial](./serial) confiuration allows to run k-means without any concurrent processing. On my own machine, a runtime for 100 observations takes ~2.7 ms. 

### MPI

*not yet implemented*

### 50-50

*not yet implemented*

### CUDA

*not yet implemented*

## Data Generation

This directory contains the data files used, as well as the Python scripts to generate the data.

These programs do nothing fancy. Data is generated by scikit-learn's [make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html). The data is 5-dimensional, so visualizing it is a bit difficult. In any case, the first two features of the generated data are plotted. Plots are generated by [matplotlib](https://matplotlib.org/3.2.1/index.html).

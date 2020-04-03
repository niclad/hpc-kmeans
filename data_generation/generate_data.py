# -*- coding: utf-8 -*-
"""
This file generates data
"""
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from pathlib import Path
import random as rng
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


MAX_SAMPLES = 10 # will need to change 100,000
FEATURES = 5 # will need to change to 5
N_CLUSTERS = 2 # will change to 3

X1, Y1 = make_blobs(n_samples=MAX_SAMPLES, n_features=FEATURES, random_state=12345678)

plt.subplot(211)
plt.scatter(X1[Y1==0, 0], X1[Y1==0, 1], marker='o', color='blue')
plt.scatter(X1[Y1==1, 0], X1[Y1==1, 1], marker='o', color='red')
plt.scatter(X1[Y1==2, 0], X1[Y1==2, 1], marker='o', color='green')

kmeans = KMeans(n_clusters=3, random_state=0, n_init=50).fit(X1)
Y2 = kmeans.labels_
plt.subplot(212)
plt.scatter(X1[Y2==0, 0], X1[Y2==0, 1], marker='o', color='blue')
plt.scatter(X1[Y2==1, 0], X1[Y2==1, 1], marker='o', color='red')
plt.scatter(X1[Y2==2, 0], X1[Y2==2, 1], marker='o', color='green')

def generate_csv(data, labels):
    file_name = "test_data_" + str(FEATURES) + "D_" + str(MAX_SAMPLES) + ".csv"
    data_file = Path('./' + file_name)
    print(file_name)
    print(Path(data_file).parent.absolute())
    if data_file.is_file():
        print("{} already exists. CSV file will not be overwritten!".format(file_name))
    #else:
    with open(file_name, "w") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",", quotechar="|", 
                                quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        for aRow in range(X1.shape[0]):
            data_row = [data[aRow, 0], data[aRow, 1], data[aRow, 2], data[aRow, 3], data[aRow, 4], labels[aRow]]
            #print(data_row[aRow])
            filewriter.writerow(data_row)
        
generate_csv(X1, Y1)
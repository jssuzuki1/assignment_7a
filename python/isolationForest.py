# Python program for running isolation forest on MNIST training images

# tested under Python 3.9.13 
# isolation forest documentation:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

import math
import numpy as np
import pandas as pd
import time
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

import getMNIST

num_runs = 100
total_elapsed_time = 0

for _ in range(num_runs):
    start_time = time.time()

    print("Running Python isolation forest . . . ")

    # isolation forest hyperparameters
    CONTAMINATION_SET = "auto"  # frequency of anomalies is unknown
    MAX_SAMPLES = "auto"  # small number ensures better results
    # set to min of 256 or training sample size
    N_ESTIMATORS = 100  # another default, number of trees in ensemble
    # max_features will be set as recommended for random forests,
    #   the square root of the number of features in the training data: 
    #   max_features = math.trunc(training.shape[1]**0.5)

    # Pandas training data frame
    training = pd.DataFrame(getMNIST.images784)

    # build anomaly detection model on training data
    SEED = np.random.RandomState(9999)
    iforest_pipeline_fit = Pipeline([
        ('scale', MinMaxScaler()),
        ('iforest', IsolationForest(max_samples = MAX_SAMPLES,
            n_estimators = N_ESTIMATORS, 
            contamination = CONTAMINATION_SET,
            max_features = math.trunc(training.shape[1]**0.5),
            random_state = SEED 
        ))
    ]).fit(training)

    # predict anomalies on training data
    # IsolationForest returns 1 = normal, -1 = anomaly
    iforestPreds = iforest_pipeline_fit.predict(training)
    iforestPythonScore = (-1)*iforest_pipeline_fit.score_samples(training)
    iforestPythonScoreDF = pd.DataFrame(iforestPythonScore)  
    iforestPythonScoreDF.to_csv("../results/pythonScores.csv", 
                                index_label = True,
                                header = ["iforestPythonScore"])

    # save MNIST labels to a comma-delimited text file
    labelsDF = pd.DataFrame(getMNIST.labels)
    labelsDF.to_csv("../results/labels.csv", 
                                index_label = True,
                                header = ["digitLabel"])

    print("Finished running Python isolation forest . . . ")
    print(training)

    end_time = time.time()
    elapsed_time = end_time - start_time
    total_elapsed_time += elapsed_time

# Calculate the elapsed time
average_elapsed_time = total_elapsed_time / num_runs
formatted_time = str(datetime.timedelta(seconds=average_elapsed_time))

print(f"Average elapsed time over {num_runs} runs: {formatted_time}")

# Open the file for writing
with open("../results/python_runtime.txt", "w") as file:
    file.write(formatted_time)
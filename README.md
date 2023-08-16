## Isolation Forest Python Vs. Go

Github Link: [insert here] 

Go is fast. Our objective is to see whether Go can generate the same results in a faster time than Python.

The "jump-start-mnist-iforest-main" folder contains both Python and Go programs.

## Brief Overview of the Isolation Forest

The Isolation Forest is an unsupervised learing algorithm that detects anomalous data patterns. The high-level idea is that, by partitioning the data repeatedly on a random subset of data on each iteration, the algorithm hones in on the "isolated" points. Generally, a smaller number of partitions is required to find outliers because they are rare and different from other instances.

Its output is a list of scores assigned to each record, where the negative scores are outliers.

## Python programs

The Python programs are located in the "jump-start-mnist-iforest-main" folder, under the "python" folder. 

There are two programs in this folder: getMNIST.py and isolationForest.py.

getMNIST.py reads in the MNIST data, while the isolationForest.py calls the former program for the data, then uses the IsolationForest package from SkLearn to obtain anomaly scores for the training images.

The original jump start program can be found here: https://github.com/ThomasWMiller/jump-start-mnist-iforest. In my amended program, a loop and time recorder has been added for benchmark purposes.

## Go Programs

There are two programs in the "go" folder. 
- "main.go" is the main program. 
- "main_test.go" is a test program of the main program, which is a unit test for importing data in the form expected of MNIST data. The test should return as a success with the terminal command "go test -v"

The Go package used in this program is called "go-iforest" from e-XpertSolutions. 

The github package is located here: https://github.com/e-XpertSolutions/go-iforest.

One critical consideration is that the parameters have different names between Go and Python. 

Here's a quick dictionary equivalency for the input parameters between the programs (left side is Python and right side is Go):
- "N_ESTIMATORS" vs "treesNumber"
- "max_samples" vs "subsampleSize"
- "contamination" vs "outliers"

Note that the equivalencies above reflect my personal understanding-- they may be incorrect.

A one-to-one equivalency in setting the parameters doesn't appear to be possible, as "max_samples" and "contamination" can have "auto" as their value. The sci-kit documentation does not say how the auto values are set in Python: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

The e-XpertSolutions package does not have "auto" as an option, which means that we will have set these options manually, leading to different results. 

## Runtime Comparison.

For a benchmark test, we loop through reading in the data set and running an isolation forest on it. The average runtimes are stored in go_average_runtime.txt and python_average_runtime.txt. 

In the case of Python, looping through this process 100 times yields an average runtime of 6.798282 seconds on my laptop. 

In the case of Go, looping through this process 100 times yields an average runtime of approximately 0.656 seconds on my laptop., which is more than 10 times faster than its Python counterpart.

However, the results between the two programs are different in their numbers, likely due to the different hyperparameters set between the two programs, which I haven't figured out how to make equivalent.

## Management Recommendation.

There is a spectre haunting anomaly detection-- the spectre of isolation forests in Go... It might be time for la vieille garde
to switch sides to the new young blood on the block. Down with the snake and up with the gopher!
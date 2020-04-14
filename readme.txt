*** HPC K-Means Evaluation ***

The code in this project aims to evaluate heterogenous computing in k-means.

Assuming that thwe directory hierarchy is maintained, all code should compile
and run without issue.

COMPILING
Compiling can be accompilished by changing to the directory for which code
should be run. Once in that directory, simply using the bash command `make`
will compile the program in question.

RUNNING
Script files are provided with all the implementations. Using sbatch with
a script file will run the code. Sbatch will return an output file
and the code will generate cluster assignments for the data as well as 
timing statistics (in the form of a CSV file).

OBSERVATION COUNT
The number of observations to cluster is coded in the C files. To change the
amount of observations clusters, change the variable `OBSERVATIONS` to either
one of 100, 1000, 10000, 100000, and 1000000. The code will have to be recompiled.
#include <iostream>
#include "kmeans.h"
#include "readData.h"
#include <string>
#include "tools.h"
#include "mpi.h"

using namespace std;
using namespace dataRead;
using namespace tools;
using namespace kmeans;

const int OBSERVATIONS = 100; // the number of data points in the file
const int FEATURES = 5;       // the number of data features
const int CLUSTERS = 3;       // the number of clusters in the data set (K)
float **x;                    // data points
const int MAX_ITER = 1000;    // maximum no. of iterations before giving up

void viewMeans(float *mu, int nFeatures, int nClusters, int iterN)
{
    cout << "Mu for iter " << iterN << endl;
    for (int i = 0; i < nFeatures * nClusters; i++)
    {
        cout << mu[i] << " ";
    }
    cout << endl;
}

void viewSets(int *sets, int nObs, int iterN)
{
    cout << "Sets for iter " << iterN << endl;
    for (int i = 0; i < nObs; i++)
    {
        cout << sets[i] << endl;
    }
}

int main(int argc, char *argv[])
{
    // mpi variables
    int size, rank; // the number of processes and process id

    // control variables
    double start, finish, total = 0; // timing values
    bool convergence = false;        // state of set convergence
    int currIter = 0;                // the current iteration for update loop

    // data variables
    float *mu;                                                            // set means
    int *labels;                                                          // gnd truth labels
    string fileName = "test_data_5D_" + to_string(OBSERVATIONS) + ".csv"; // data file name
    int *sets;                                                            // the observations in each set
    int *prevSets;                                                        // the set from the previous iteration - F

    // initialize mpi
    MPI_Init(&argc, &argv);


    x = new float *[OBSERVATIONS]; // initialize memory on the host
    labels = new int[OBSERVATIONS];
    sets = new int[OBSERVATIONS];
    prevSets = new int[OBSERVATIONS];
    mu = new float[FEATURES * CLUSTERS];

    // initialize x
    for (int i = 0; i < OBSERVATIONS; i++)
    {
        x[i] = new float[FEATURES];
    }

    //initialize mu
    for (int i = 0; i < CLUSTERS * FEATURES; i++)
        mu[i] = 0;

    // ===== READ DATA =====
    start = CLOCK();
    readData(fileName, x, labels, FEATURES); // read the data from the data file
    finish = CLOCK() - start;
    total += finish;
    cout << "File read time: " << finish << " msec." << endl;

    printSample(5, x, labels, FEATURES); // print first 5 oberservations and labels read from input file

    // ===== INITIALIZE K-MEANS =====
    start = CLOCK();
    forgy(CLUSTERS, FEATURES, mu, x, OBSERVATIONS); // initialize means using Forgy method
    // viewMeans(mu, FEATURES, CLUSTERS, -1);          // DEBUGGING

    updateSets(sets, CLUSTERS, x, mu, OBSERVATIONS, FEATURES); // initialize sets by updating the assigned values
    // viewSets(sets, 10, -1);                                    // DEBUGGING

    finish = CLOCK() - start;
    total += finish;
    cout << "K-means init. time: " << finish << " msec." << endl;

    // ===== OPERATE ON SETS =====
    start = CLOCK();
    while (!convergence && (currIter < MAX_ITER))
    {
        arrayCopy(OBSERVATIONS, prevSets, sets);                   // update prevSets, to keep last sets
        updateSets(sets, CLUSTERS, x, mu, OBSERVATIONS, FEATURES); // update the sets with the new means
        for (int aSet = 0; aSet < CLUSTERS; aSet++)                // update the set means with a new set elements
            setsMean(aSet, sets, x, mu, OBSERVATIONS, FEATURES);

        // viewMeans(mu, FEATURES, CLUSTERS, currIter); // DEBUGGING
        // viewSets(sets, 10, currIter);      // DEBUGGING

        convergence = arrayCompare(OBSERVATIONS, prevSets, sets); // check the current and previous sets for convergence
        currIter++;
    }
    finish = CLOCK() - start;
    total += finish;

    // display convergence statistics
    if (currIter < MAX_ITER - 1)
        cout << "Convergence reached in " << currIter + 1 << " iterations." << endl;
    else
        cout << "Maximum iterations reached. Convergence status unknown." << endl;

    cout << "1 iter. time: " << finish << " msec." << endl;

    // ===== SAVE FINAL LABELS =====
    start = CLOCK();
    saveData("out_data.csv", sets, OBSERVATIONS); // save the updated labels
    finish = CLOCK() - start;
    total += finish;
    cout << "File save time: " << finish << " msec." << endl;

    cout << "Total operation time: " << total << " msec." << endl;

    for (int i = 0; i < 10; i++) // free host memory
        delete[] x[i];
    delete[] x;
    delete[] labels;
    delete[] sets;
    delete[] prevSets;
    delete[] mu;
}
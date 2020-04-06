#include <iostream>
#include "kmeans.h"
#include "mpi.h"
#include "readData.h"
#include <string>
#include "tools.h"

using namespace std;
using namespace dataRead;
using namespace tools;
using namespace kmeans;

const int OBSERVATIONS = 10000;     // the number of data points in the file
const int FEATURES = 5;           // the number of data features
const int CLUSTERS = 3;           // the number of clusters in the data set (K)
float x[OBSERVATIONS][FEATURES];  // data points -- MUST BE CONTIGUOUS DATA
float mu[FEATURES * CLUSTERS];    // means for the cluster sets
const int MAX_ITER = 1000;        // maximum no. of iterations before giving up
const int npp = OBSERVATIONS / 4; // numPerProc--should always 0.25 * OBSERVATIONS

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
    int rank, size;   // the process id and number of processes
    double maxFinish; // the maximum running time for the world

    // control variables
    double start, finish, total = 0; // timing values
    char convergence = 0;            // state of set convergence
    int currIter = 0;                // the current iteration for update loop
    double timingStats[6];           // would likely work better as a struct

    string fileName = "test_data_5D_" + to_string(OBSERVATIONS) + ".csv"; // data file name
    string statsName = "running_stats-" + to_string(OBSERVATIONS) + ".csv";

    // data variables
    int labels[OBSERVATIONS];        // gnd truth labels
    int sets[OBSERVATIONS];          // the observations in each set
    int prevSets[OBSERVATIONS];      // the set from the previous iteration - F
    int counts[CLUSTERS];            // the counts for the elements in each set
    float sums[FEATURES * CLUSTERS]; // the sums of all the elements in each set

    // ===== SIZE VARIABLES =====

    int procSets[npp];      // sets assigned to a proc
    int procPrev[npp];      // prev sets assigned to a proc
    float tempMu[FEATURES]; // temporary mu for procs updating mean
    int tempCounts[CLUSTERS];
    float tempSums[FEATURES * CLUSTERS];

    // initialize mu
    for (int i = 0; i < CLUSTERS * FEATURES; i++)
        mu[i] = 0;

    // initialize mpi
    MPI_Init(&argc, &argv); // start mpi
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // initialize arrays that count
    initArray<int>(CLUSTERS, counts);
    initArray<int>(CLUSTERS, tempCounts);
    initArray<float>(CLUSTERS * FEATURES, sums);
    initArray<float>(CLUSTERS * FEATURES, tempSums);

    // ===== READ DATA =====
    MPI_Barrier(MPI_COMM_WORLD);
    start = CLOCK();
    if (rank == 0) // only let head proc. read data.
    {
        readData(fileName, x, labels, FEATURES); // read the data from the data file
        //printSample(5, x, labels, FEATURES); // print first 5 oberservations and labels read from input file - DEBUGGING
    }
    MPI_Bcast(x, OBSERVATIONS * FEATURES, MPI_FLOAT, 0, MPI_COMM_WORLD); // send observations to all processes
    finish = (CLOCK() - start);
    MPI_Reduce(&finish, &maxFinish, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // get longest running time

    if (rank == 0)
    {
        total += maxFinish;
        cout << "File read time: " << maxFinish << " msec." << endl;
        timingStats[0] = maxFinish;
    }
    // ===== INITIALIZE K-MEANS =====
    MPI_Barrier(MPI_COMM_WORLD);
    start = CLOCK();
    if (rank == 0)                                      // only let one process handle this
        forgy(CLUSTERS, FEATURES, mu, x, OBSERVATIONS); // initialize means using Forgy method
    // viewMeans(mu, FEATURES, CLUSTERS, -1);          // DEBUGGING

    MPI_Bcast(mu, CLUSTERS * FEATURES, MPI_FLOAT, 0, MPI_COMM_WORLD); // send the means to all the processes

    // for (int i = 0; i < size; i++)
    // {
    //     if (rank == i)
    //     {
    //         cout << "Rank " << i << ": ";
    //         for (int j = 0; j < 5; j++)
    //             cout << x[25][j] << " ";
    //         cout << endl;
    //     }
    // }

    updateSets(procSets, CLUSTERS, tempCounts, tempSums, x, mu, OBSERVATIONS, FEATURES, rank, npp); // initialize sets by updating the assigned values
    // viewSets(sets, 10, -1);                                    // DEBUGGING

    // for (int i = 0; i < size; i++)
    // {
    //     if (rank == i)
    //     {
    //         cout << "Rank " << i << ": ";
    //         for (int j = 0; j < 25; j++)
    //             cout << procSets[j] << " ";
    //         cout << endl;
    //     }
    // }

    // make sure all processes have the updated sets and relevant information
    MPI_Gather(procSets, npp, MPI_INT, sets, npp, MPI_INT, 0, MPI_COMM_WORLD);              // gather the set assignments
    MPI_Allreduce(tempCounts, counts, CLUSTERS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);          // sum all counts
    MPI_Allreduce(tempSums, sums, FEATURES * CLUSTERS, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); // sum all sums

    //MPI_Barrier(MPI_COMM_WORLD);

    if (rank < 3)
    {
        setsMean(rank, sets, x, tempMu, OBSERVATIONS, FEATURES, counts, sums, rank);

        // for (int i = 0; i < FEATURES; i++)
        //     cout << tempMu[i] << " ";
        // cout << endl;
    }

    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(tempMu, FEATURES, MPI_FLOAT, mu, FEATURES, MPI_FLOAT, 0, MPI_COMM_WORLD); // get all the means
    MPI_Bcast(mu, FEATURES * CLUSTERS, MPI_FLOAT, 0, MPI_COMM_WORLD);                    // distribute the means

    finish = (CLOCK() - start);
    cout << "RANK " << rank << ": " << finish << " msec." << endl;
    MPI_Reduce(&finish, &maxFinish, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // get longest running time

    if (rank == 0)
    {
        total += maxFinish;
        cout << "K-means init. time: " << maxFinish << " msec." << endl;
        timingStats[1] = maxFinish;
    }

    // ===== OPERATE ON SETS =====
    MPI_Barrier(MPI_COMM_WORLD);
    start = CLOCK();
    MPI_Scatter(sets, npp, MPI_INT, procSets, npp, MPI_INT, 0, MPI_COMM_WORLD);
    currIter = 0;
    while (!convergence && (currIter < MAX_ITER))
    {
        // reset proc based counts and sums
        initArray<int>(CLUSTERS, tempCounts);
        initArray<float>(CLUSTERS * FEATURES, tempSums);

        if (rank == 0) // have head node copy the set for later comparison
            arrayCopy(OBSERVATIONS, prevSets, sets);

        updateSets(procSets, CLUSTERS, tempCounts, tempSums, x, mu, OBSERVATIONS, FEATURES, rank, npp);

        MPI_Reduce(tempCounts, counts, CLUSTERS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(tempSums, sums, CLUSTERS * FEATURES, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank <= 3)
            setsMean(rank, sets, x, tempMu, OBSERVATIONS, FEATURES, counts, sums, rank);

        //MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(procSets, npp, MPI_INT, sets, npp, MPI_INT, 0, MPI_COMM_WORLD);           // update the sets
        MPI_Gather(tempMu, FEATURES, MPI_FLOAT, mu, FEATURES, MPI_FLOAT, 0, MPI_COMM_WORLD); // get all the means
        MPI_Bcast(mu, FEATURES * CLUSTERS, MPI_FLOAT, 0, MPI_COMM_WORLD);                    // distribute the means

        // if (currIter < 2)
        // {
        //     for (int i = 0; i < 4; i++)
        //     {
        //         if (rank == i)
        //         {
        //             cout << rank << endl;
        //             for (int i = 0; i < 15; i++)
        //             {
        //                 cout << mu[i] << " ";
        //             }
        //             cout << endl;
        //             MPI_Barrier(MPI_COMM_WORLD);
        //         }
        //     }
        // }

        if (rank == 0)
            convergence = arrayCompare(OBSERVATIONS, prevSets, sets); // check the current and previous sets for convergence

        //MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&convergence, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        currIter++;
    }
    MPI_Gather(procSets, npp, MPI_INT, sets, npp, MPI_INT, 0, MPI_COMM_WORLD);

    finish = (CLOCK() - start);
    MPI_Reduce(&finish, &maxFinish, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // get longest running time
    //MPI_Reduce(&currIter, &currIter, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);   // get the most number of iterations

    if (rank == 0)
    {
        total += maxFinish;

        // display convergence statistics
        if (currIter >= MAX_ITER)
            cout << "Maximum iterations reached. Convergence status unknown." << endl;

        cout << "Convergence time: " << maxFinish << " msec. in " << currIter << " iterations." << endl;
        timingStats[2] = maxFinish;
        timingStats[3] = currIter;
    }

    // ===== SAVE FINAL LABELS =====
    if (rank == 0)
    {
        start = CLOCK();
        saveData("out_data.csv", sets, OBSERVATIONS); // save the updated labels
        finish = CLOCK() - start;
        total += finish;
        cout << "File save time: " << finish << " msec." << endl;
        timingStats[4] = finish;
        cout << "Total operation time: " << total << " msec." << endl;
        timingStats[5] = total;

        // save running statistics
        saveStats(timingStats, statsName);
        cout << "END" << endl;
    }

    MPI_Finalize();
}
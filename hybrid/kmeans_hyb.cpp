#include <iostream>
#include <mpi.h>
#include <string>
#include "tools.h"

using namespace std;
using namespace tools;

const int OBSERVATIONS = 100;    // the number of data points in the file
const int FEATURES = 5;           // the number of data features
const int CLUSTERS = 3;           // the number of clusters in the data set (K)
const int MAX_ITER = 10000;       // maximum no. of iterations before giving up
const int npp = OBSERVATIONS / 4; // numPerProc--should always 0.25 * OBSERVATIONS

extern void dispDevice();
extern void updateSetsWrapper(float *x, float *mu, float *sums, int *counts, int *sets, int nSets, int nFeatures, int nObs, int rank);
extern void muWrapper(float *mu, float *sums, int *counts, int nFeatures, int nSets);
extern void copyWrapper(int *sets, int *prevSets, int nObs);
extern void convergeWrapper(int *sets, int *prevSets, bool *converge, int nObs);
extern void testWrapper();

int main(int argc, char *argv[])
{
    // MPI variables
    int rank, size;   // proc. id and no. of procs
    double maxFinish; // largest running time for a section

    // control variables
    double start, finish, total = 0; // timing stats
    char convergence = 0;            // state of set convergence
    bool worldConv;
    int currIter = 0; // the current iteration for the update loop
    double timingStats[6];

    string fileName = "test_data_5D_" + to_string(OBSERVATIONS) + ".csv";   // data file name
    string statsName = "running_stats-" + to_string(OBSERVATIONS) + ".csv"; // stats file name

    // data variables
    float x[FEATURES * OBSERVATIONS]; // the data as a 1D array
    float mu[FEATURES * CLUSTERS];    // set means
    int sets[OBSERVATIONS];           // observation set assignments
    int prevSets[OBSERVATIONS];       // previous iterations set assignments
    int counts[CLUSTERS];             // the number of elements assigned to each set
    float sums[FEATURES * CLUSTERS];  // the sum of the elements in each set
    bool converge[OBSERVATIONS];

    // initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get the number of procs.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get the proc's id

    // MPI arrays
    int procSets[npp];
    int procPrev[npp];
    float tempMu[FEATURES];
    int tempCounts[CLUSTERS];
    float tempSums[FEATURES * CLUSTERS];
    bool tempConv[npp];

    // initialize arrays with 0
    initArray<float>(FEATURES * CLUSTERS, mu);
    initArray<float>(FEATURES * CLUSTERS, sums);
    initArray<float>(FEATURES * CLUSTERS, tempSums);
    initArray<int>(CLUSTERS, counts);
    initArray<int>(CLUSTERS, tempCounts);

    if (rank == 0)
        dispDevice();

    // ===== READ DATA =====
    MPI_Barrier(MPI_COMM_WORLD);
    start = CLOCK();
    if (rank == 0)
    {
        readData(fileName, x, FEATURES); // read the data from the data file
    }
    MPI_Bcast(x, OBSERVATIONS * FEATURES, MPI_FLOAT, 0, MPI_COMM_WORLD); // distribute data to all processes
    finish = (CLOCK() - start);
    MPI_Reduce(&finish, &maxFinish, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // get the longest running time

    if (rank == 0)
    {
        total += maxFinish;
        cout << "File read time: " << maxFinish << " msec." << endl;
        timingStats[0] = maxFinish;
    }

    // for (int i = 0; i < size; i++)
    // {
    //     if (rank == i)
    //     {
    //         for (int j = 0; j < 5; j++)
    //         {
    //             cout << x[j] << " ";
    //         }
    //         cout << endl;
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    // ===== INITIALIZE K-MEANS =====
    MPI_Barrier(MPI_COMM_WORLD);
    start = CLOCK();
    if (rank == 0)
    {
        forgy(CLUSTERS, FEATURES, mu, x, OBSERVATIONS); // get 3 random points to be initial means
        finish = CLOCK() - start;
        cout << "forgy: " << finish << " msec." << endl;
    }
    MPI_Bcast(mu, CLUSTERS * FEATURES, MPI_FLOAT, 0, MPI_COMM_WORLD); // distribute the means to all the processes
    double uswS, uswE, uswM;

    uswS = CLOCK();
    updateSetsWrapper(x, mu, tempSums, tempCounts, procSets, CLUSTERS, FEATURES, npp, rank); // initialize sets based on initial means
    uswE = CLOCK() - uswS;

    MPI_Gather(procSets, npp, MPI_INT, sets, npp, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Allreduce(tempCounts, counts, CLUSTERS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(tempSums, sums, FEATURES * CLUSTERS, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // if (rank == 0)
    // {

    //     for (int i = 0; i < 3; i++)
    //         cout << counts[i] << " ";
    //     cout << endl;

    //     // for (int i = 0; i < 15; i++)
    //     //     cout << sums[i] << " ";
    //     // cout << endl;
    // }

    if (rank == 0)
    {
        double umwS = CLOCK();
        muWrapper(mu, sums, counts, FEATURES, CLUSTERS);
        double umwE = CLOCK() - umwS;
        cout << "UMW time: " << umwE << " msec." << endl;
    }

    MPI_Bcast(mu, FEATURES * CLUSTERS, MPI_FLOAT, 0, MPI_COMM_WORLD); // broadcast updated means

    // if (rank == 2)
    // {
    //     for (int i = 0; i < 15; i++)
    //         cout << mu[i] << " ";
    //     cout << endl;
    // }

    finish = CLOCK() - start;
    MPI_Reduce(&finish, &maxFinish, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // get the longest running time
    MPI_Reduce(&uswE, &uswM, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // get max updateSets wrapper time

    if (rank == 0)
    {
        total += maxFinish;
        cout << "K-means init. time: " << maxFinish << " msec." << endl;
        timingStats[1] = maxFinish;
        cout << "USW time: " << uswM << " msec." << endl;
    }

    // ===== OPERATE ON SETS =====
    // if (rank == 0) // save the previous set
    // {
    //     copyWrapper(sets, prevSets, OBSERVATIONS);
    //     // for (int i = 0; i < 100; i++)
    //     //     cout << sets[i] << " " << prevSets[i] << endl;

    //     convergeWrapper(sets, prevSets, converge, OBSERVATIONS);

    //     for (int i = 0; i < 100; i++)
    //         cout << converge[i] << endl;

    //     convergence = (char)arrayCompare(OBSERVATIONS, converge);
    //     cout << boolalpha << (bool)convergence << endl;
    // }

    MPI_Barrier(MPI_COMM_WORLD);
    start = CLOCK();
    MPI_Scatter(sets, npp, MPI_INT, procSets, npp, MPI_INT, 0, MPI_COMM_WORLD);
    currIter = 0; // already assigned but why not?
    while (!convergence && (currIter < MAX_ITER))
    {
        // reset proc based counts and sums
        initArray<int>(CLUSTERS, tempCounts);
        initArray<float>(CLUSTERS * FEATURES, tempSums);

        if (rank == 0) // save the previous set
            copyWrapper(sets, prevSets, OBSERVATIONS);

        uswS = CLOCK();
        updateSetsWrapper(x, mu, tempSums, tempCounts, procSets, CLUSTERS, FEATURES, npp, rank);
        uswE = CLOCK() - uswS;
        MPI_Reduce(&uswE, &uswM, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // get max updateSets wrapper time
        if (rank == 0 && currIter < 1)
            cout << "Iter. USW time: " << uswM << " msec." << endl;


        MPI_Reduce(tempCounts, counts, CLUSTERS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(tempSums, sums, CLUSTERS * FEATURES, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) // update means
            muWrapper(mu, sums, counts, FEATURES, CLUSTERS);

        MPI_Gather(procSets, npp, MPI_INT, sets, npp, MPI_INT, 0, MPI_COMM_WORLD); // update the sets
        MPI_Bcast(mu, FEATURES * CLUSTERS, MPI_FLOAT, 0, MPI_COMM_WORLD);          // distribute the means

        if (rank == 0) // check for convergence on head node
        {
            convergeWrapper(sets, prevSets, converge, OBSERVATIONS);
            convergence = (char)arrayCompare(OBSERVATIONS, converge);
        }

        // convergence = arrayCompare(npp, tempConv);

        // MPI_Reduce(&convergence, &worldConv, 1, MPI_CXX_BOOL, MPI_LAND, 0, MPI_COMM_WORLD); // get results of convergence statuses
        MPI_Bcast(&convergence, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

        currIter++;
    }
    MPI_Gather(procSets, npp, MPI_INT, sets, npp, MPI_INT, 0, MPI_COMM_WORLD);
    finish = CLOCK() - start;
    MPI_Reduce(&finish, &maxFinish, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // get longest running time

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
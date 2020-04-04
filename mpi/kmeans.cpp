#include <algorithm>
#include <cmath>
#include <iostream>
#include "kmeans.h"

using namespace std;
using namespace kmeans;

/**
 * @brief Determine the mean for a given set
 * 
 * @param setid 
 * @param sets 
 * @param x 
 * @param mu 
 * @param nObs 
 * @param nFeatures 
 */
void kmeans::setsMean(int setid, int *sets, float x[][5], float *mu, const int nObs, const int nFeatures, int *counts, float *sums, int rank)
{
    // this function relies on proc size >= nSets. Each proc will find a set's mean
    for (int j = 0; j < nFeatures; j++)
    {
        int idx = j + (nFeatures * rank);
        mu[idx] = sums[idx] / counts[rank];
    }
}

/**
 * @brief Update the sets by assigning observations 
 * based on (squared) Euclidean distances between an observation
 * and a set's mean.
 * 
 * @param sets          The list of set observation assignments
 * @param nSets         The number of sets
 * @param x             The observations
 * @param mu            The means for each set
 * @param nObs          The number of observations
 * @param nFeatures     The number of features for the observations
 */
void kmeans::updateSets(int *sets, int nSets, int *counts, float *sums, float x[][5], float *mu, const int nObs, const int nFeatures, const int rank, const int npp)
{

    int startIdx = rank * npp;       // the index the rank is allowed to start counting at
    int endIdx = (rank + 1) * npp;   // the index the rank counts up to (exclusive)
    float dists[3]; // Collection of Euclidean distances; 3 == number of sets (nSets)

    for (int i = startIdx; i < endIdx; i++)
    {
        // initialize dists to be 0 every time an centroid test is run
        for (int m = 0; m < nSets; m++)
            dists[m] = 0;

        int idx = 0; // dists index
        int j = 0;   // feautre index
        for (int k = 0; k < nFeatures * nSets; k++)
        {
            j = k % nFeatures; // add offset to j for x features

            dists[idx] += (x[i][j] - mu[k]) * (x[i][j] - mu[k]); // cumsum the current distance val

            if ((k + 1) % nFeatures == 0)
                idx++; // update dists index based on current cluster
        }
        // for (int m = 0; m < nSets; m++) cout << dists[m] << " "; cout << endl; // DEBUGGING
        int assignment = minIdx(dists, nSets);

        sets[i - startIdx] = assignment;
        counts[assignment]++;

        int idx2;
        for (int j = 0; j < nFeatures; j++) // j < nfeats
        {
            idx2 = j + (nFeatures * assignment);
            sums[idx2] += x[i][j];
        }
    }
}

/**
 * @brief Get the index for the smallest item in the given array
 * 
 * @param arr   Array to compare items within
 * @return int  Number of elements in the array
 */
int kmeans::minIdx(float *arr, int n)
{
    float currMin = arr[0];
    int currIdx = 0;

    for (int i = 1; i < n; i++)
    {
        if (arr[i] < currMin)
        {
            currMin = arr[i];
            currIdx = i;
        }
    }

    return currIdx;
}

//int muOffset(int setid, int nFeatures) { return nFeatures * setid; }
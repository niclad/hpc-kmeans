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
void kmeans::setsMean(int setid, int *sets, float **x, float *mu, const int nObs, const int nFeatures)
{
    int currSet;                      // the current set in the iteration
    int muOffset = nFeatures * setid; // offset for mu location for the given set
    int setObs = 0;                   // a count of the number of observations in the set

    for (int i = 0; i < nFeatures; i++)
        mu[i + muOffset] = 0; // initialize mu to be 0 before updating

    for (int i = 0; i < nObs; i++)
    {
        currSet = sets[i];
        if (currSet == setid)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                mu[j + muOffset] += x[i][j]; // at the observation's feature to the mean sum
            }
            setObs++;
        }
    }

    // calculate the mean
    if (setObs > 0)
    {
        for (int i = 0; i < nFeatures; i++)
        {
            mu[i + muOffset] = mu[i + muOffset] / setObs; // get the mean for this feature
            //cout << mu[i + muOffset] << endl; // DEBUGGING
        }
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
void kmeans::updateSets(int *sets, int nSets, float **x, float *mu, const int nObs, const int nFeatures)
{
    float *dists = new float[nSets]; // Collection of Euclidean distances; 3 == number of sets
    
    for (int i = 0; i < nObs; i++)
    {
        // initialize dists to be 0 every time an centroid test is run
        for (int m = 0; m < nSets; m++)
            dists[m] = 0;

        int idx = 0; // dists index
        int j = 0;
        for (int k = 0; k < nFeatures * nSets; k++)
        {
            j = k % nFeatures; // add offset to j for x features
            dists[idx] += (x[i][j] - mu[k]) * (x[i][j] - mu[k]); // cumsum the current distance val
            
            if ((k + 1) % nFeatures == 0)
                idx ++; // update dists index based on current cluster
        }
        //for (int m = 0; m < nSets; m++) cout << dists[m] << " "; cout << endl; // DEBUGGING
        sets[i] = minIdx(dists, nSets);
    }

    delete[] dists;
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
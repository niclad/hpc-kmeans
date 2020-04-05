#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <iostream>
#include "kmeans.h"

using namespace std;
using namespace kmeans;

__global__ void kmeans::updateSets(float *d_x, float *d_mu, float *d_sums, int *d_counts, int *d_sets, int nSets, int nFeatures, int nObs)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
    if (t_id >= nObs) return;

    float currObs[5];
    int startIdx = t_id * nFeatures;
    
    for (int i = 0; i < nFeautres; i++)
    {
        currObs[i] = d_x[t_id + i];
    }

    float bestDist = FLT_MAX; // maximum float
    int bestSet = 

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
#include <cuda.h>
#include <cfloat>
#include <iostream>
#include "readData.h"
#include <string>
#include "tools.h"

using namespace std;
using namespace dataRead;
using namespace tools;

const int OBSERVATIONS = 100; // the number of data points in the file
const int FEATURES = 5;       // the number of data features
const int CLUSTERS = 3;       // the number of clusters in the data set (K)
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

__device__ float
dist(float *currObs, float *currMu, int nFeatures)
{
    float distance = 0;
    for (int i = 0; i < nFeatures; i++)
    {
        distance += (currObs[i] - currMu[i]) * (currObs[i] - currMu[i]);
    }
    return distance;
}

__global__ void updateSets(float *d_x, float *d_mu, float *d_sums, int *d_counts, int *d_sets, int nSets, int nFeatures, int nObs)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
    if (t_id >= nObs)
        return;

    float currObs[5];
    float currMu[5];
    int startIdx = t_id * nFeatures;

    for (int i = 0; i < nFeatures; i++)
    {
        currObs[i] = d_x[t_id + i];
    }

    float bestDist = FLT_MAX; // maximum float
    int bestSet = 0;
    for (int aSet = 0; aSet < nSets; aSet++)
    {
        for (int i = 0; i < nFeatures; i++)
        {
            currMu[i] = d_mu[(aSet * nFeatures) + i];
        }

        float distance = dist(currObs, currMu, nFeatures);
        if (distance < bestDist)
        {
            bestDist = distance;
            bestSet = aSet;
        }
    }
    d_sets[t_id] = bestSet;
    atomicAdd(&d_counts[bestSet], 1);

    for (int i = 0; i < nFeatures; i++)
    {
        atomicAdd(&d_sums[(bestSet * nFeatures) + i], currObs[i]);
    }
}

__global__ void computeMu(float *d_mu, float *d_sums, int *d_counts, int nFeatures)
{
    int set = threadIdx.x;
    int offset = set * nFeatures;
    int counts = 1;
    if (d_counts[set] != 0)
        counts = d_counts[set];
    for (int i = 0; i < nFeatures; i++)
    {
        d_mu[offset + i] = d_sums[offset + i] / counts;
    }
}

__global__ void copySets(int *d_sets, int *d_prevSets, int nObs)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread id

    if (t_id >= nObs)
        return;

    d_prevSets[t_id] = d_sets[t_id];
}

__global__ void checkConvergence(int *d_sets, int *d_prevSets, bool *d_converge, int nObs)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // get global thread id

    if (t_id > nObs)
        return;

    d_converge[t_id] = d_sets[t_id] == d_prevSets[t_id];
}

int main()
{
    // control variables
    double start, finish, total = 0; // timing values
    bool convergence = false;            // state of set convergence
    int currIter = 0;                // the current iteration for update loop
    double timingStats[6];

    // data variables
    float *h_x, *h_mu, *h_sums; // host data
    int *h_sets, *h_prevSets, *h_counts;
    float *d_x, *d_mu, *d_sums; // device data
    int *d_sets, *d_prevSets, *d_counts;
    bool *d_converge, *h_converge; // convergence check

    // size variables
    size_t obsBytes, muBytes, setsBytes, cntBytes, convBytes;
    obsBytes = sizeof(float) * OBSERVATIONS * FEATURES; // get the sizes in bytes of device arrays
    muBytes = sizeof(float) * FEATURES * CLUSTERS;
    setsBytes = sizeof(int) * OBSERVATIONS;
    cntBytes = sizeof(int) * CLUSTERS;
    convBytes = sizeof(char) * OBSERVATIONS;

    h_x = new float[OBSERVATIONS * FEATURES]; // allocate memory on host
    h_mu = new float[FEATURES * CLUSTERS];
    h_sums = new float[FEATURES * CLUSTERS];
    h_sets = new int[OBSERVATIONS];
    h_prevSets = new int[OBSERVATIONS];
    h_counts = new int[CLUSTERS];
    h_converge = new bool[OBSERVATIONS];

    cudaMalloc(&d_x, obsBytes); // allocate memory on device
    cudaMalloc(&d_mu, muBytes);
    cudaMalloc(&d_sums, muBytes);
    cudaMalloc(&d_sets, setsBytes);
    cudaMalloc(&d_prevSets, setsBytes);
    cudaMalloc(&d_counts, cntBytes);
    cudaMalloc(&d_converge, convBytes);

    string fileName = "test_data_5D_" + to_string(OBSERVATIONS) + ".csv"; // data file name

    // initialize mu and h_sums
    for (int i = 0; i < CLUSTERS * FEATURES; i++)
    {
        h_mu[i] = 0;
        h_sums[i] = 0;
    }

    // initialize counts
    for (int i = 0; i < CLUSTERS; i++)
        h_counts[i] = 0;

    // get device properties
    cudaDeviceProp props;               // devices properties
    cudaGetDeviceProperties(&props, 0); // get the device properties
    cout << "GPU: " << props.name << ": " << props.major << "." << props.minor << endl;

    // ===== READ DATA =====
    start = CLOCK();
    readData(fileName, h_x, FEATURES); // read the data from the data file
    finish = CLOCK() - start;
    total += finish;
    cout << "File read time: " << finish << " msec." << endl;
    timingStats[0] = finish;
    //printSample(5, x, labels, FEATURES); // print first 5 oberservations and labels read from input file - DEBUGGING

    cudaMemcpy(d_x, h_x, obsBytes, cudaMemcpyHostToDevice);           // copy observations to device
    cudaMemcpy(d_sums, h_sums, muBytes, cudaMemcpyHostToDevice);      // copy sums to device
    cudaMemcpy(d_counts, h_counts, cntBytes, cudaMemcpyHostToDevice); // copy counts to device

    // ===== INITIALIZE K-MEANS =====
    start = CLOCK();
    forgy(CLUSTERS, FEATURES, h_mu, h_x, OBSERVATIONS); // initialize means on host

    for (int i = 0; i < 10; i++)
    {
        cout << h_x[i] << endl;
    }

    cudaMemcpy(d_mu, h_mu, muBytes, cudaMemcpyHostToDevice); // copy means to device

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)OBSERVATIONS / blockSize);

    updateSets<<<gridSize, blockSize>>>(d_x, d_mu, d_sums, d_counts, d_sets, CLUSTERS, FEATURES, OBSERVATIONS); // initialize sets by updating the assigned values

    computeMu<<<1, CLUSTERS>>>(d_mu, d_sums, d_counts, FEATURES); // update means based on new sets;

    finish = CLOCK() - start;
    total += finish;
    cout << "K-means init. time: " << finish << " msec." << endl;
    timingStats[1] = finish;

    // ===== OPERATE ON SETS =====
    start = CLOCK();
    while (!convergence && (currIter < MAX_ITER))
    {
        copySets<<<gridSize, blockSize>>>(d_sets, d_prevSets, OBSERVATIONS);                                       // update prevSets, to keep last sets
        updateSets<<<gridSize, blockSize>>>(d_x, d_mu, d_sums, d_counts, d_sets, CLUSTERS, FEATURES, OBSERVATIONS); // update the sets with the new means

        computeMu<<<1, CLUSTERS>>>(d_mu, d_sums, d_counts, FEATURES);

        // viewMeans(mu, FEATURES, CLUSTERS, currIter); // DEBUGGING
        // viewSets(sets, 10, currIter);      // DEBUGGING

        checkConvergence<<<gridSize, blockSize>>>(d_sets, d_prevSets, d_converge, OBSERVATIONS); // check the current and previous sets for convergence
        cudaMemcpy(h_converge, d_converge, convBytes, cudaMemcpyDeviceToHost);

        convergence = arrayCompare(OBSERVATIONS, h_converge); // check if everything is equal
        currIter++;
    }
    cudaMemcpy(h_sets, d_sets, setsBytes, cudaMemcpyDeviceToHost);

    finish = CLOCK() - start;
    total += finish;

    // display convergence statistics
    if (currIter >= MAX_ITER)
        cout << "Maximum iterations reached. Convergence status unknown." << endl;

    cout << "Convergence time: " << finish << " msec. in " << currIter << " iterations." << endl;
    timingStats[2] = finish;
    timingStats[3] = currIter;

    // ===== SAVE FINAL LABELS =====
    start = CLOCK();
    saveData("out_data.csv", h_sets, OBSERVATIONS); // save the updated labels
    finish = CLOCK() - start;
    total += finish;
    cout << "File save time: " << finish << " msec." << endl;
    timingStats[4] = finish;
    cout << "Total operation time: " << total << " msec." << endl;
    timingStats[5] = total;

    // save running statistics
    string statsName = "running_stats-" + to_string(OBSERVATIONS) + ".csv";
    saveStats(timingStats, statsName);

    cudaFree(d_x); // free device memory
    cudaFree(d_mu);
    cudaFree(d_sums);
    cudaFree(d_sets);
    cudaFree(d_prevSets);
    cudaFree(d_counts);
    cudaFree(d_converge);

    delete[] h_x; // free host memory
    delete[] h_mu;
    delete[] h_sums;
    delete[] h_sets;
    delete[] h_prevSets;
    delete[] h_counts;
    delete[] h_converge;
}
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <iostream>

using namespace std;

/**
 * @brief Print the device's properties
 * 
 */
extern void dispDevice()
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    cout << "GPU: " << props.name << ": " << props.major << "." << props.minor << endl;
}

__device__ float 
dist(float *currObs, float *currMu, int nFeatures)
{
    float distance = 0;
    for (int i = 0; i < nFeatures; i++)
        distance += (currObs[i] - currMu[i]) * (currObs[i] - currMu[i]);

    return distance;
}

__global__ void updateSets(float *x, float *mu, float *sums, int *counts, int *sets, int nSets, int nFeatures, int nObs)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // get global thread ID
    if (t_id >= nObs)
        return;

    float currObs[5];
    float currMu[5];

    for (int i = 0; i < nFeatures; i++)
        currObs[i] = x[t_id + i];

    float bestDist = FLT_MAX; // maximum representable float
    int bestSet = 0;
    for (int aSet = 0; 0 < nSets; aSet++)
    {
        for (int i = 0; i < nFeatures; i++)
            currMu[i] = mu[(aSet * nFeatures) + i];

        float distance = dist(currObs, currMu, nFeatures);
        if (distance < bestDist)
        {
            bestDist = distance;
            bestSet = aSet;
        }
    }
    sets[t_id] = bestSet;
    atomicAdd(&counts[bestSet], 1);

    for (int i = 0; i < nFeatures; i++)
        atomicAdd(&sums[(bestSet * nFeatures) + i], currObs[i]);
}

extern void updateSetsWrapper(float *x, float *mu, float *sums, int *counts, int *sets, int nSets, int nFeatures, int nObs)
{
    float *d_x, *d_mu, *d_sums;
    int *d_sets, *d_prevSets, *d_counts;
    bool *d_converge;

    size_t obsBytes, muBytes, setsBytes, cntBytes, convBytes;
    obsBytes = sizeof(float) * nObs * nFeatures; // get the sizes in bytes of device arrays
    muBytes = sizeof(float) * nFeatures * nSets;
    setsBytes = sizeof(int) * nObs;
    cntBytes = sizeof(int) * nSets;
    convBytes = sizeof(char) * nObs;

    cudaMalloc(&d_x, obsBytes); // allocate memory on device
    cudaMalloc(&d_mu, muBytes);
    cudaMalloc(&d_sums, muBytes);
    cudaMalloc(&d_sets, setsBytes);
    cudaMalloc(&d_prevSets, setsBytes);
    cudaMalloc(&d_counts, cntBytes);
    cudaMalloc(&d_converge, convBytes);

    cudaMemcpy(d_x, x, obsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sums, sums, muBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, counts, cntBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu, mu, muBytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)nObs / blockSize);

    updateSets<<<gridSize, blockSize>>>(d_x, d_mu, d_sums, d_counts, d_sets, nSets, nFeatures, nObs);

    cudaMemcpy(x, d_x, obsBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sums, d_sums, muBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(counts, d_counts, cntBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(mu, d_mu, muBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_mu);
    cudaFree(d_sums);
    cudaFree(d_sets);
    cudaFree(d_prevSets);
    cudaFree(d_counts);
    cudaFree(d_converge);
}

__global__ void computeMu(float *mu, float *sums, int *counts, int nFeatures)
{
    int set = threadIdx.x;
    int offset = set * nFeatures;
    int i_counts = 1;
    if(counts[set] != 0)
        i_counts = counts[set];
    for(int i = 0; i < nFeatures; i++)
        mu[offset + i] = sums[offset + i] / i_counts;
}

extern void muWrapper(float *mu, float *sums, int *counts, int nFeatures, int nSets)
{
    float *d_mu, *d_sums;
    int *d_counts;

    size_t muBytes, cntBytes;
    muBytes = sizeof(float) * nFeatures * nSets;
    cntBytes = sizeof(int) * nSets;

    cudaMalloc(&d_mu, muBytes);
    cudaMalloc(&d_sums, muBytes);
    cudaMalloc(&d_counts, cntBytes);

    cudaMemcpy(d_mu, mu, muBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sums, sums, muBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, counts, cntBytes, cudaMemcpyHostToDevice);

    computeMu<<<1, 3>>>(d_mu, d_sums, d_counts, nFeatures);

    cudaMemcpy(mu, d_mu, muBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_mu);
    cudaFree(d_sums);
    cudaFree(d_counts);
}

__global__ void copySets(int *sets, int *prevSets, int nObs)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // get global thread id

    if (t_id >= nObs)
        return;
    
    prevSets[t_id] = sets[t_id];
}

extern void copyWrapper(int *sets, int *prevSets, int nObs)
{
    int *d_sets, *d_prevSets;
    size_t setsBytes= sizeof(int) * nObs;

    cudaMalloc(&d_sets, setsBytes);
    cudaMalloc(&d_prevSets, setsBytes);

    cudaMemcpy(d_sets, sets, setsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prevSets, prevSets, setsBytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)nObs / blockSize);

    copySets<<<gridSize, blockSize>>>(d_sets, d_prevSets, nObs);

    cudaMemcpy(prevSets, d_prevSets, setsBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_sets);
    cudaFree(d_prevSets);
}

__global__ void checkConvergence(int *sets, int *prevSets, bool *converge, int nObs)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // get global thread id

    if (t_id >= nObs)
        return;
    
    converge[t_id] = sets[t_id] == prevSets[t_id];
}

extern void convergeWrapper(int *sets, int *prevSets, bool *converge, int nObs)
{
    int *d_sets, *d_prevSets;
    bool *d_converge;
    size_t setsBytes, convBytes;
    setsBytes = sizeof(int) * nObs;
    convBytes = sizeof(bool) * nObs;

    cudaMalloc(&d_sets, setsBytes);
    cudaMalloc(&d_prevSets, setsBytes);
    cudaMalloc(&d_converge, convBytes);

    cudaMemcpy(d_sets, sets, setsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prevSets, prevSets, setsBytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)nObs / blockSize);

    checkConvergence<<<gridSize, blockSize>>>(d_sets, d_prevSets, d_converge, nObs);

    cudaMemcpy(converge, d_converge, setsBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_sets);
    cudaFree(d_prevSets);
    cudaFree(d_converge);
}
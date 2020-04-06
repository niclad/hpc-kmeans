#include <cmath>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
    if (t_id < n)
        c[t_id] = a[t_id] + b[t_id];

    if (t_id == 0)
        printf("GPU works\n");
}

extern void wrapper(double *c)
{
    printf("Initializing GPU data...\n");
    int n = 100000;                    // Size of vectors
    double *h_a, *h_b;                 // Host input vectors
    double *h_c;                       // Host output vector
    double *d_a, *d_b;                 // Device input vectors
    double *d_c;                       // Device output vector
    size_t bytes = n * sizeof(double); // Size, in bytes, of each vector

    h_a = new double[n]; // allocate memory for vectors on host
    h_b = new double[n];
    h_c = new double[n];
    cudaMalloc(&d_a, bytes); // allocate memory for vectors on device
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    for (int i = 0; i < n; i++) // initialize host vectors
    {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
    blockSize = 1024; // number of threads in each block
    gridSize = (int)ceil((float)n / blockSize); // number of thread blocks in a grid

    printf("... Initialized.\nExecuting kernel...\n");

    // execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("... Kernel executed\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}

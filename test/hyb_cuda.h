#ifndef _HYB_CUDA_
#define _HYB_CUDA_

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vecAdd(double *a, double *b, double *c, int n);
void wrapper(double *c);

#endif
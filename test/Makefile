# based on this Makefile: https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-aware-mpi-example/src/Makefile
CUDALDFLAGS=-L${CUDA_INSTALL_PATH}/lib64 -lcudart
MPICFLAGS=-I${MPI_HOME}/include
CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include

all: program

hybrid.o: hybrid.cpp
	mpic++ -std=c++11 -c hybrid.cpp -o hybrid.o

hyb_cuda.o: hyb_cuda.cu
	nvcc -std=c++11 -c hyb_cuda.cu -o hyb_cuda.o

program: hybrid.o hyb_cuda.o
	mpic++ -std=c++11 hybrid.o hyb_cuda.o -lcudart -L/apps/CUDA/cuda-9.0/lib64 -o program
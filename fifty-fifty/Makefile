all: kmeans_hyb

kmeans_hyb.o: kmeans_hyb.cpp
	mpic++ -std=c++11 -c kmeans_hyb.cpp -o kmeans_hyb.o

tools.o: tools.cpp tools.h
	mpic++ -std=c++11 -c tools.cpp -o tools.o

cuda_control.o: cuda_control.cu
	nvcc -std=c++11 -c cuda_control.cu -o cuda_control.o

kmeans_hyb: kmeans_hyb.o tools.o cuda_control.o
	mpic++ -std=c++11 kmeans_hyb.o tools.o cuda_control.o -lcudart -L/apps/CUDA/cuda-9.0/lib64 -o kmeans_hyb
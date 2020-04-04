#include <iostream>
#include "mpi.h"

int main(int argc, char *argv[])
{
    int size, rank;
    int test[5] = {1};
    std::cout << rank << ": ";
    for (int i = 0; i < 5; i++)
        std::cout << test[i] << " ";
    std::cout << std::endl;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Finalize();    
}
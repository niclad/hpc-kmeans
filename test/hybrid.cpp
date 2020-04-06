// hybrid cuda-mpi implementation test

#include <mpi.h>
#include <stdio.h>
#include <cmath>

extern void wrapper(double *c);

int main(int argc, char* argv[])
{
    int size, rank; // mpi # procs and proc id

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Rank %d is a go!\n", rank);

    if (rank == 0)
    {
        int n = 100000;
        double *c = new double[n];
        wrapper(c);

        double sum = 0;
        for (int i = 0; i < n; i++)
            sum += c[i];

        printf("Final result: %f\n", sum / n); // should be 1.0

        delete[] c;
    }    

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank != 0)
    {
        printf("Rank %d has nothing to do.\n", rank);
    }

    MPI_Finalize();
}
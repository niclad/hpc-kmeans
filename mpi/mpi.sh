#!/bin/bash
#
#SBATCH	--job-name=mpi_test
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1
#SBATCH --output=mpi_test-%j.out
#SBATCH --time=5:00

mpirun --mca btl_base_warn_component_unused 0 ./kmeans_mpi

# for value in {1..10}
# do
# 	echo "RUN: " $value
# 	./kmeans_serial
# 	echo "*****************************"
# done

# echo "END RUN"

#!/bin/bash
#
#SBATCH	--job-name=cuda_test
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1
#SBATCH --output=km_cuda-%j.out
#SBATCH --time=5:00

for value in {1..10}
do
	echo "RUN: " $value
	./kmeans_cuda
	echo "*****************************"
done

echo "END RUN"

#!/bin/bash
#
#SBATCH	--job-name=ser-10k
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1
#SBATCH --output=serial_10000-%j.out
#SBATCH --time=5:00

for value in {1..10}
do
	echo "RUN: " $value
	./kmeans_serial
	echo "*****************************"
done

echo "END RUN"

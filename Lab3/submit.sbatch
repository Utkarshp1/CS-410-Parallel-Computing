#!/bin/bash
#SBATCH --job-name=MPI_JOB    	      # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=email@iitdh.ac.in # Where to send mail	
#SBATCH --nodes=2                     # Run on 2 nodes
#SBATCH --ntasks-per-node=32          # 32 per node
#SBATCH --cpus-per-task=1             # 1 cpu per task
#SBATCH --time=12:10:00               # Time limit hrs:min:sec
#SBATCH --output=test_%j.log   # Standard output and error log
#SBATCH --partition=test   # Queuename
#SBATCH -n 64	# Must be a perfect square

pwd; hostname; date
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
srun --n 4 ./summa 6912
srun --n 9 ./summa 6912
srun --n 16 ./summa 6912
srun --n 25 ./summa 6912
srun --n 36 ./summa 6912
srun --n 49 ./summa 6912
srun --n 64 ./summa 6912
srun --n 4 ./summa 6912
srun --n 9 ./summa 6912
srun --n 16 ./summa 6912
srun --n 25 ./summa 6912
srun --n 36 ./summa 6912
srun --n 49 ./summa 6912
srun --n 64 ./summa 6912
srun --n 4 ./summa 6912
srun --n 9 ./summa 6912
srun --n 16 ./summa 6912
srun --n 25 ./summa 6912
srun --n 36 ./summa 6912
srun --n 49 ./summa 6912
srun --n 64 ./summa 6912
srun --n 4 ./summa 6912
srun --n 9 ./summa 6912
srun --n 16 ./summa 6912
srun --n 25 ./summa 6912
srun --n 36 ./summa 6912
srun --n 49 ./summa 6912
srun --n 64 ./summa 6912
date


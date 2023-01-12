#!/bin/bash
#SBATCH --job-name=GPU_JOB    	      # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=email@iitdh.ac.in # Where to send mail	
#SBATCH --nodes=1                     # Run on a single node
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
pwd; hostname; date

set -x
srun make exp1
srun make exp1
srun make exp1

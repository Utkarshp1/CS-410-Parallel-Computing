#!/bin/bash
#SBATCH --job-name=CILK_JOB    	      # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=email@iitdh.ac.in # Where to send mail	
#SBATCH --nodes=1                     # Run on a single node
#SBATCH --ntasks-per-node=1           # Single task
#SBATCH --cpus-per-task=32            # 32 CPUs per task
#SBATCH --time=04:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
pwd; hostname; date

set -ex
make exp1
make exp2
make exp3
date

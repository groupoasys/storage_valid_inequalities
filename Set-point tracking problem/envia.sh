#!/usr/bin/env bash
# The name to show in queue lists for this job:
#SBATCH -J set_tracking_problem

# Number of desired cores (all in same node):
#SBATCH --cpus-per-task=1

# Amount of RAM needed for this job:
#SBATCH --mem=2gb

# The time the job will be running (four models x 100 batteries per task):
#SBATCH --time=0-00:30:00

# * to request only machines with 16 cores and 64GB with InfiniBand network
#SBATCH --constraint=cal

# Set output and error files
#SBATCH --error=zzz.%A.%a.err
#SBATCH --output=zzz.%A.%a.out

# Array job: one task per PV profile (1..200)
#SBATCH --array=1-200

sleep $SLURM_ARRAY_TASK_ID
module purge
module load python/3.11.4       # Python with amplpy installed (~/.local/lib/python3.11)
module load ampl/20260520       # AMPL with its Gurobi and MOSEK drivers and licenses

# the program to execute with its parameters:
hostname
time python setpoint_tracking_letter.py

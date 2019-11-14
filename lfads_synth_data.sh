#!/bin/bash
#SBATCH --qos=high                              # Ask for unkillable job
#SBATCH --cpus-per-task=1                       # Ask for 1 CPUs
#SBATCH --gres=gpu:1                            # Ask for 1 GPU
#SBATCH --mem=10G                               # Ask for 10 GB of RAM
#SBATCH --time=5:00:00                          # The job will run for 5 hours
#SBATCH -o /network/tmp1/princelu/slurm-%j.out  # Write the log on tmp1

while getopts d:s:m: option
do
case "${option}"
d) SYSTEM=${OPTARG};;
s) SEED=${OPTARG};;
m) MODEL=${OPTARG}
esac
done

module purge
module load python/3.7
source pytorch/bin/activate

python generate_synthetic_data -d $SYSTEM -s $SEED -o $SLURM_TMPDIR
python run_lfads -d $SLURM_TMPDIR/synth_data/${SYSTEM}_${SEED} -p parameters/parameters_${SYSTEM}_${MODEL}.yaml -o $SLURM_TMPDIR

cp -r $SLURM_TMPDIR/models $HOME/hierarchical_lfads/models

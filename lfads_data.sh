#!/bin/bash
#SBATCH --qos=high                              # Ask for unkillable job
#SBATCH --cpus-per-task=1                       # Ask for 1 CPUs
#SBATCH --gres=gpu:1                            # Ask for 1 GPU
#SBATCH --mem=10G                               # Ask for 10 GB of RAM
#SBATCH --time=5:00:00                          # The job will run for 5 hours
#SBATCH -o /network/tmp1/princelu/slurm-%j.out  # Write the log on tmp1

start=`date +%s`
module purge
module load python/3.7
source $HOME/pytorch/bin/activate

echo $DATA
echo $PARAMETERS

mkdir $SLURM_TMPDIR/data
mkdir $SLURM_TMPDIR/models

cp $DATA $SLURM_TMPDIR/data

python run_lfads.py -d $SLURM_TMPDIR/$DATA -p $PARAMETERS -o $SLURM_TMPDIR

cp -r $SLURM_TMPDIR/models $HOME/hierarchical_lfads

end=`date +%s`
runtime=$((end-start))
echo "Runtime was $runtime seconds"

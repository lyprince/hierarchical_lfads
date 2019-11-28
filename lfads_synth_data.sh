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

echo $SYSTEM
echo $SEED
echo $MODEL

mkdir $SLURM_TMPDIR/synth_data
mkdir $SLURM_TMPDIR/models

if [ -e $HOME/synth_data/${SYSTEM}_${SEED} ]
then
  cp $HOME/synth_data/${SYSTEM}_${SEED} $SLURM_TMPDIR/synth_data/${SYSTEM}_${SEED}
else
  python generate_synthetic_data.py -d $SYSTEM -s $SEED -o $SLURM_TMPDIR -p $HOME/hierarchical_lfads/synth_data/${SYSTEM}_params.yaml
fi

python run_lfads_synth.py -d $SLURM_TMPDIR/synth_data/${SYSTEM}_${SEED} -p parameters/parameters_${SYSTEM}_${MODEL}.yaml -o $SLURM_TMPDIR

cp -r $SLURM_TMPDIR/models $HOME/hierarchical_lfads
cp -r $SLURM_TMPDIR/synth_data $HOME/hierarchical_lfads

end=`date +%s`
runtime=$((end-start))
echo "Runtime was $runtime seconds"

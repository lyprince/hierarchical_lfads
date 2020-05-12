#!/bin/bash
#SBATCH --output ../out/chaotic-rnn_%A.out # Write out
#SBATCH --error ../out/chaotic-rnn_%A.err  # Write error
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --job-name=chaotic-rnn
#SBATCH --mem=10GB
#SBATCH --time=48:00:00

module purge
module load python/3.7
source $HOME/pytorch/bin/activate

exit_script(){
cp -r $SLURM_TMPDIR/ $PROJECTDIR
}

terminator(){
echo 'job killed'
}

echo $SEED

PROJECTDIR=$HOME/hierarchical_lfads
DATADIR=$PROJECTDIR/synth_data
HPDIR=$PROJECTDIR/hyperparameters/chaotic-rnn
OUTDIR=$PROJECTDIR/models

python $PROJECTDIR/generate_synthetic_data.py -d chaotic-rnn -o $DATADIR -p $DATADIR/chaotic-rnn_params.yaml -s $SEED
DATAPATH=$PROJECTDIR/synth_data/chaotic-rnn_$SEED

python $PROJECTDIR/preprocessing_oasis.py -d $DATAPATH -t 0.3 -s 1.0 -k
python $PROJECTDIR/preprocessing_oasis.py -d $DATAPATH -t 0.3 -s 4.0
OK=ok_t0.3_s1.0
OU=ou_t0.3_s4.0
DATAPATH_OK=$DATAPATH'_'$OK
DATAPATH_OU=$DATAPATH'_'$OU

LFADS_MODEL_DESC=cenc128_cont128_fact20_genc128_gene200_glat64_ulat1_orion-/
SVLAE_MODEL_DESC=dcen128_dcon128_dgen128_dgla64_dula1_fact20_gene200_ocon64_oenc64_olat100_orion-/
SEED_OK=$SEED'_'$OK
SEED_OU=$SEED'_'$OU

python $PROJECTDIR/train_model.py -m lfads -d $DATAPATH -p $HPDIR/lfads.yaml -o $OUTDIR --batch_size 200 --max_epochs 2000 --data_suffix spikes
python $PROJECTDIR/train_model.py -m lfads -d $DATAPATH_OK -p $HPDIR/lfads.yaml -o $OUTDIR --batch_size 200 --max_epochs 2000 --data_suffix ospikes
python $PROJECTDIR/train_model.py -m svlae -d $DATAPATH_OK -p $HPDIR/svlae.yaml -o $OUTDIR --batch_size 200 --max_epochs 2000 --data_suffix fluor
python $PROJECTDIR/train_model.py -m lfads -d $DATAPATH_OU -p $HPDIR/lfads.yaml -o $OUTDIR --batch_size 200 --max_epochs 2000 --data_suffix ospikes
python $PROJECTDIR/train_model.py -m svlae -d $DATAPATH_OU -p $HPDIR/svlae.yaml -o $OUTDIR --batch_size 200 --max_epochs 2000 --data_suffix 

python $PROJECTDIR/infer_latent.py -m $OUTDIR/chaotic-rnn_$SEED/lfads/$LFADS_MODEL_DESC -d $DATAPATH --data_suffix spikes
python $PROJECTDIR/infer_latent.py -m $OUTDIR/chaotic-rnn_$SEED_OK/lfads_oasis/$LFADS_MODEL_DESC -d $DATAPATH_OK --data_suffix ospikes
python $PROJECTDIR/infer_latent.py -m $OUTDIR/chaotic-rnn_$SEED_OK/svlae/$SVLAE_MODEL_DESC -d $DATAPATH_OK --data_suffix fluor
python $PROJECTDIR/infer_latent.py -m $OUTDIR/chaotic-rnn_$SEED_OU/lfads_oasis/$LFADS_MODEL_DESC -d $DATAPATH_OU --data_suffix ospikes
python $PROJECTDIR/infer_latent.py -m $OUTDIR/chaotic-rnn_$SEED_OU/svlae/$SVLAE_MODEL_DESC -d $DATAPATH_OU --data_suffix fluor
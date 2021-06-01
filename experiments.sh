#!/bin/bash

### This script contains the experiments
### The script is based on Computecanada with my account, make sure to change the directories and settings based on your machine
## VP models use checkpoint 8 which is the correct one to replicate Yang Song paper best results


# Sync your files
rsync --exclude='.git/' -avz --no-g --no-p /home/jolicoea/my_projects/score_sde_faster_sampling $SLURM_TMPDIR
cd $SLURM_TMPDIR/score_sde_faster_sampling

# Load the libraries
module load python/3.8.2 StdEnv/2020 gcc/9.3.0 cuda/11.0 cudnn/8.0.3 scipy-stack/2020b julia/1.6.0
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip setuptools
# Flax 0.3.1 has a bug, need 0.3.0 but then there is a new bug, new version after should have it fixed (haven't tested it yet)
pip install --upgrade numpy==1.19.2 tensorflow_gpu tensorboard tensorflow-gan tensorflow_io tensorflow_datasets tensorflow-addons absl-py flax==0.3.0 jax==0.2.8 ml_collections torch 

# Install jaxlib (make sure to adjust with the correct settings)
python3 jaxlibprep.py -V 0.1.59 -C cuda110 -P cp38 --set-runpath ${CUDA_PATH}/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2020/CUDA/cuda11.0/cudnn/8.0.3/lib64 -t linux
pip install --upgrade tmp/repacked/*.whl

# Options needed for Jax
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_PATH}/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2020/CUDA/cuda11.0/cudnn/8.0.3/lib64
export CUDA_DIR=${CUDA_PATH}; 
export JAX_OMNISTAGING=1;
export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_PATH} 

cp -r "/scratch/jolicoea/fid_stats/." "$SLURM_TMPDIR/score_sde_faster_sampling/assets/stats" # transfer your fid_stats (you will create those)
cp -r "/scratch/jolicoea/checkpoints/." "$SLURM_TMPDIR/score_sde_faster_sampling" # transfer your checkpoints (grab them from https://drive.google.com/drive/folders/10pQygNzF7hOOLwP3q8GiNxSnFRpArUxQ?usp=sharing)

export total_lowres=50000
export bs_lowres=900 # batch-size is chosen assuming 64Gb of GPU RAM (2 V-100 with 32Gb)
export total_highres=5000 # Not the full FID, feel free to change, but be prepared to wait a long time!
export bs_highres=100 # batch-size is chosen assuming 128Gb of GPU RAM (4 V-100 with 32Gb)


###### Create the FID statistic for each dataset (do once)
# CIFAR-10
python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'fid_stats' --workdir 'fid_stats_cifar10' --config.eval.batch_size=128
cp assets/stats/cifar10_32_stats.npz /scratch/jolicoea/fid_stats/cifar10_32_stats.npz # copy to permanent storage

# Bedroom (256x256)
python main.py --config 'configs/ncsnpp/church.py' --mode 'fid_stats' --workdir 'fid_stats_church' --config.eval.batch_size=128
cp assets/stats/lsun_256_stats.npz /scratch/jolicoea/fid_stats/lsun_church_256_stats.npz # copy to permanent storage

# FFHQ (256x256)
python main.py --config 'configs/ncsnpp/ffhq_256.py' --mode 'fid_stats' --workdir 'fid_stats_ffhq' --config.eval.batch_size=128
cp assets/stats/ffhq_256_stats.npz /scratch/jolicoea/fid_stats/ffhq_256_stats.npz # copy to permanent storage


###### 32 x 32 (Table 1)

## VP-deep
python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="reverse_diffusion" --config.sampling.corrector="langevin" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres
python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="ddim" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2
python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=330 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres
python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="ddim" --config.sampling.corrector="none" --config.model.num_scales=330 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0078 --config.sampling.reltol=2e-2
python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=274 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres
python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="ddim" --config.sampling.corrector="none" --config.model.num_scales=274 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0078 --config.sampling.reltol=5e-2
python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=180 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres
python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="ddim" --config.sampling.corrector="none" --config.model.num_scales=180 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-1
python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=151 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres
python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.predictor="ddim" --config.sampling.corrector="none" --config.model.num_scales=151 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ddpmpp/cifar10_deep_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous_deep' --config.sampling.method="ode" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

## VP
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="reverse_diffusion" --config.sampling.corrector="langevin" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="ddim" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=329 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="ddim" --config.sampling.corrector="none" --config.model.num_scales=329 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0078 --config.sampling.reltol=2e-2
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=274 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="ddim" --config.sampling.corrector="none" --config.model.num_scales=274 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0078 --config.sampling.reltol=5e-2
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=179 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="ddim" --config.sampling.corrector="none" --config.model.num_scales=179 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-1
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=147 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.predictor="ddim" --config.sampling.corrector="none" --config.model.num_scales=147 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.method="ode" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres


## VE
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.predictor="reverse_diffusion" --config.sampling.corrector="langevin" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0039 --config.sampling.reltol=1e-2
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=738 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0039 --config.sampling.reltol=2e-2
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=490 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0039 --config.sampling.reltol=5e-2
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=271 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0039 --config.sampling.reltol=1e-1
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=170 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.method="ode" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres


## VE-deep
python main.py --config 'configs/ncsnpp/cifar10_deep_continuous_ve.py' --mode 'eval' --workdir 'cifar10_ncsnpp_deep_continuous' --config.sampling.predictor="reverse_diffusion" --config.sampling.corrector="langevin" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ncsnpp/cifar10_deep_continuous_ve.py' --mode 'eval' --workdir 'cifar10_ncsnpp_deep_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ncsnpp/cifar10_deep_continuous_ve.py' --mode 'eval' --workdir 'cifar10_ncsnpp_deep_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0039 --config.sampling.reltol=1e-2
python main.py --config 'configs/ncsnpp/cifar10_deep_continuous_ve.py' --mode 'eval' --workdir 'cifar10_ncsnpp_deep_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=736 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ncsnpp/cifar10_deep_continuous_ve.py' --mode 'eval' --workdir 'cifar10_ncsnpp_deep_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0039 --config.sampling.reltol=2e-2
python main.py --config 'configs/ncsnpp/cifar10_deep_continuous_ve.py' --mode 'eval' --workdir 'cifar10_ncsnpp_deep_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=488 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ncsnpp/cifar10_deep_continuous_ve.py' --mode 'eval' --workdir 'cifar10_ncsnpp_deep_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0039 --config.sampling.reltol=5e-2
python main.py --config 'configs/ncsnpp/cifar10_deep_continuous_ve.py' --mode 'eval' --workdir 'cifar10_ncsnpp_deep_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=270 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ncsnpp/cifar10_deep_continuous_ve.py' --mode 'eval' --workdir 'cifar10_ncsnpp_deep_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres --config.sampling.abstol=0.0039 --config.sampling.reltol=1e-1
python main.py --config 'configs/ncsnpp/cifar10_deep_continuous_ve.py' --mode 'eval' --workdir 'cifar10_ncsnpp_deep_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=170 --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres

python main.py --config 'configs/ncsnpp/cifar10_deep_continuous_ve.py' --mode 'eval' --workdir 'cifar10_ncsnpp_deep_continuous' --config.sampling.method="ode" --config.sampling.corrector="none" --config.eval.num_samples=$total_lowres --config.eval.batch_size=$bs_lowres


###### 256 x 256  (Table 2)

## Church experiments
python main.py --config 'configs/ncsnpp/church.py' --mode 'eval' --workdir 'church_ncsnpp_continuous' --config.sampling.predictor="reverse_diffusion" --config.sampling.corrector="langevin" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

python main.py --config 'configs/ncsnpp/church.py' --mode 'eval' --workdir 'church_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

python main.py --config 'configs/ncsnpp/church.py' --mode 'eval' --workdir 'church_ncsnpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres --config.sampling.abstol=0.0039 --config.sampling.reltol=1e-2
python main.py --config 'configs/ncsnpp/church.py' --mode 'eval' --workdir 'church_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=1104 --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

python main.py --config 'configs/ncsnpp/church.py' --mode 'eval' --workdir 'church_ncsnpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres --config.sampling.abstol=0.0039 --config.sampling.reltol=2e-2
python main.py --config 'configs/ncsnpp/church.py' --mode 'eval' --workdir 'church_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=1030 --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

python main.py --config 'configs/ncsnpp/church.py' --mode 'eval' --workdir 'church_ncsnpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres --config.sampling.abstol=0.0039 --config.sampling.reltol=5e-2
python main.py --config 'configs/ncsnpp/church.py' --mode 'eval' --workdir 'church_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=648 --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

python main.py --config 'configs/ncsnpp/church.py' --mode 'eval' --workdir 'church_ncsnpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres --config.sampling.abstol=0.0039 --config.sampling.reltol=1e-1
python main.py --config 'configs/ncsnpp/church.py' --mode 'eval' --workdir 'church_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=201 --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

python main.py --config 'configs/ncsnpp/church.py' --mode 'eval' --workdir 'church_ncsnpp_continuous' --config.sampling.method="ode" --config.sampling.corrector="none" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

## FFHQ experiments
python main.py --config 'configs/ncsnpp/ffhq_256.py' --mode 'eval' --workdir 'ffhq_256_ncsnpp_continuous' --config.sampling.predictor="reverse_diffusion" --config.sampling.corrector="langevin" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

python main.py --config 'configs/ncsnpp/ffhq_256.py' --mode 'eval' --workdir 'ffhq_256_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

python main.py --config 'configs/ncsnpp/ffhq_256.py' --mode 'eval' --workdir 'ffhq_256_ncsnpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres --config.sampling.abstol=0.0039 --config.sampling.reltol=1e-2
python main.py --config 'configs/ncsnpp/ffhq_256.py' --mode 'eval' --workdir 'ffhq_256_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=1020 --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

python main.py --config 'configs/ncsnpp/ffhq_256.py' --mode 'eval' --workdir 'ffhq_256_ncsnpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres --config.sampling.abstol=0.0039 --config.sampling.reltol=2e-2
python main.py --config 'configs/ncsnpp/ffhq_256.py' --mode 'eval' --workdir 'ffhq_256_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=643 --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

python main.py --config 'configs/ncsnpp/ffhq_256.py' --mode 'eval' --workdir 'ffhq_256_ncsnpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres --config.sampling.abstol=0.0039 --config.sampling.reltol=5e-2
python main.py --config 'configs/ncsnpp/ffhq_256.py' --mode 'eval' --workdir 'ffhq_256_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=336 --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

python main.py --config 'configs/ncsnpp/ffhq_256.py' --mode 'eval' --workdir 'ffhq_256_ncsnpp_continuous' --config.sampling.predictor="adaptive" --config.sampling.corrector="none" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres --config.sampling.abstol=0.0039 --config.sampling.reltol=1e-1
python main.py --config 'configs/ncsnpp/ffhq_256.py' --mode 'eval' --workdir 'ffhq_256_ncsnpp_continuous' --config.sampling.predictor="euler_maruyama" --config.sampling.corrector="none" --config.model.num_scales=198 --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres

python main.py --config 'configs/ncsnpp/ffhq_256.py' --mode 'eval' --workdir 'ffhq_256_ncsnpp_continuous' --config.sampling.method="ode" --config.sampling.corrector="none" --config.eval.num_samples=$total_highres --config.eval.batch_size=$bs_highres


##### Appendix

## CIFAR-10: VE - normal hyperparameter experiments to highlight best parameters

export total=10000
export bs=900 # batch-size is chosen assuming 64Gb of GPU RAM (2 V-100 with 32Gb)

# default
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=True --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.9 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# use_prev=False
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=True --config.sampling.error_use_prev=False  --config.sampling.safety=0.9  --config.sampling.exp=0.9 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# sde_improved_euler=False, extrapolation=True, exp=0.50
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=False --config.sampling.extrapolation=True --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.5 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# sde_improved_euler=False, extrapolation=False, exp=0.50
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=False --config.sampling.extrapolation=False --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.5 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# sde_improved_euler=False, extrapolation=False, exp=0.50, norm="Linf"
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=False --config.sampling.extrapolation=False --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.5 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2 --config.sampling.norm="Linf"

# sde_improved_euler=False, extrapolation=False, exp=0.50, norm="Linf" safety=.8
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=False --config.sampling.extrapolation=False --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.8  --config.sampling.exp=0.5 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2 --config.sampling.norm="Linf"

# extrapolation=False
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=False --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.9 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# norm="Linf"
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=True --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.9 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2 --config.sampling.norm="Linf"

# exp=0.5
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=True --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.5 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# exp=0.8
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=True --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=.8 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# exp=1
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ncsnpp/cifar10_continuous_ve.py' --mode 'eval' --workdir 've_cifar10_ncsnpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=True --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=1 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2


## CIFAR-10: VP - normal hyperparameter experiments to highlight best parameters

export total=10000
export bs=900 # batch-size is chosen assuming 64Gb of GPU RAM (2 V-100 with 32Gb)

# default
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=True --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.9 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# use_prev=False
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=True --config.sampling.error_use_prev=False  --config.sampling.safety=0.9  --config.sampling.exp=0.9 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# sde_improved_euler=False, extrapolation=True, exp=0.50
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=False --config.sampling.extrapolation=True --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.5 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# sde_improved_euler=False, extrapolation=False, exp=0.50
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=False --config.sampling.extrapolation=False --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.5 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# sde_improved_euler=False, extrapolation=False, exp=0.50, norm="Linf"
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=False --config.sampling.extrapolation=False --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.5 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2 --config.sampling.norm="Linf"

# sde_improved_euler=False, extrapolation=False, exp=0.50, norm="Linf" safety=.8
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=False --config.sampling.extrapolation=False --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.8  --config.sampling.exp=0.5 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2 --config.sampling.norm="Linf"

# extrapolation=False
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=False --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.9 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# norm="Linf"
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=True --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.9 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2 --config.sampling.norm="Linf"

# exp=0.5
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=True --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=0.5 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# exp=0.8
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=True --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=.8 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

# exp=1
rm -r vp_cifar10_ddpmpp_continuous/eval
python main.py --config 'configs/ddpmpp/cifar10_continuous_vp_c8.py' --mode 'eval' --workdir 'vp_cifar10_ddpmpp_continuous' --config.sampling.corrector="none" --config.sampling.adaptive=True --config.eval.num_samples=$total --config.eval.batch_size=$bs --config.sampling.sde_improved_euler=True --config.sampling.extrapolation=True --config.sampling.error_use_prev=True --config.sampling.error_use_other=True  --config.sampling.safety=0.9  --config.sampling.exp=1 --config.sampling.abstol=0.0078 --config.sampling.reltol=1e-2

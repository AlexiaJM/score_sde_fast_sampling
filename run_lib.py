# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
from typing import Any

import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
import functools
from flax.metrics import tensorboard
from flax.training import checkpoints
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
import utils
from models import utils as mutils
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags

FLAGS = flags.FLAGS


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  rng = jax.random.PRNGKey(config.seed)
  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  if jax.host_id() == 0:
    writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  rng, step_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(step_rng, config)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(checkpoint_meta_dir)
  # Resume training when intermediate checkpoints are detected
  state = checkpoints.restore_checkpoint(checkpoint_meta_dir, state)
  # `state.step` is JAX integer on the GPU/TPU devices
  initial_step = int(state.step)
  rng = state.rng

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              additional_dim=config.training.n_jitted_steps,
                                              uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, score_model, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  # Pmap (and jit-compile) multiple training steps together for faster running
  p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
  eval_step_fn = losses.get_step_fn(sde, score_model, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)
  # Pmap (and jit-compile) multiple evaluation steps together for faster running
  p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size // jax.local_device_count(), config.data.image_size,
                      config.data.image_size, config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

  # Replicate the training state to run on multiple devices
  pstate = flax_utils.replicate(state)
  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  if jax.host_id() == 0:
    logging.info("Starting training loop at step %d." % (initial_step,))
  rng = jax.random.fold_in(rng, jax.host_id())

  # JIT multiple training steps together for faster training
  n_jitted_steps = config.training.n_jitted_steps
  # Must be divisible by the number of steps jitted together
  assert config.training.log_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
         config.training.eval_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

  for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))  # pylint: disable=protected-access
    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
    next_rng = jnp.asarray(next_rng)
    # Execute one training step
    (_, pstate), ploss = p_train_step((next_rng, pstate), batch)
    loss = flax.jax_utils.unreplicate(ploss).mean()
    # Log to console, file and tensorboard on host 0
    if jax.host_id() == 0 and step % 50 == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss))
      writer.scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and jax.host_id() == 0:
      saved_state = flax_utils.unreplicate(pstate)
      saved_state = saved_state.replace(rng=rng)
      checkpoints.save_checkpoint(checkpoint_meta_dir, saved_state,
                                  step=step // config.training.snapshot_freq_for_preemption,
                                  keep=1)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), next(eval_iter))  # pylint: disable=protected-access
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      (_, _), peval_loss = p_eval_step((next_rng, pstate), eval_batch)
      eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
      if jax.host_id() == 0:
        logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss))
        writer.scalar("eval_loss", eval_loss, step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      if jax.host_id() == 0:
        saved_state = flax_utils.unreplicate(pstate)
        saved_state = saved_state.replace(rng=rng)
        checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                    step=step // config.training.snapshot_freq,
                                    keep=np.inf)

      # Generate and save samples
      if config.training.snapshot_sampling:
        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
        sample, n = sampling_fn(sample_rng, pstate)
        this_sample_dir = os.path.join(
          sample_dir, "iter_{}_host_{}".format(step, jax.host_id()))
        tf.io.gfile.makedirs(this_sample_dir)
        image_grid = sample.reshape((-1, *sample.shape[2:]))
        nrow = int(np.sqrt(image_grid.shape[0]))
        sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
          np.save(fout, sample)

        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
          utils.save_image(image_grid, fout, nrow=nrow, padding=2)


def evaluate(config, workdir, eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to "eval".
  """

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)  
  # Build data pipeline
  #train_ds, eval_ds, _ = datasets.get_dataset(config,
  #                                            additional_dim=1,
  #                                            uniform_dequantization=config.data.uniform_dequantization,
  #                                            evaluation=True)

  _, eval_ds, _ = datasets.get_dataset(config, additional_dim=1, evaluation=True,
                                       uniform_dequantization=config.data.uniform_dequantization)
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  rng = jax.random.PRNGKey(config.seed + 1)
  rng, model_rng = jax.random.split(rng)

  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state, ema_rate=config.model.ema_rate,
                       params_ema=initial_params, rng=rng)  # pytype: disable=wrong-keyword-args

  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  bpd_num_repeats, ds_bpd = create_likelihood_dataloaders(config)

  def save_checkpoint(eval_meta, step):
    checkpoints.save_checkpoint(eval_dir, eval_meta, step=step, keep=1, prefix=f"meta_{jax.host_id()}_")

  if config.eval.enable_loss:
    p_eval_step = setup_p_eval_step(config, sde, score_model)

  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, score_model, inverse_scaler)

  if config.eval.enable_sampling:
    bs = config.eval.batch_size // jax.local_device_count()
    imsize = config.data.image_size
    sampling_shape = (bs, imsize, imsize, config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = jax.random.fold_in(rng, jax.host_id())

  @flax.struct.dataclass
  class EvalMeta:
    """ A data class for storing intermediate results to resume evaluation after pre-emption. """
    ckpt_id: int
    sampling_round_id: int
    bpd_round_id: int
    rng: Any

  # Add one additional round if necessary since we round down
  num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + (config.eval.num_samples % config.eval.batch_size > 0)
  try: ## Alexia: We had to do this because FFHQ doesn't have a dataset length!
    num_bpd_rounds = len(ds_bpd) * bpd_num_repeats
  except:
    num_bpd_rounds = 999999

  # Restore evaluation after pre-emption
  eval_meta = EvalMeta(config.eval.begin_ckpt, -1, -1, rng)
  eval_meta = checkpoints.restore_checkpoint(eval_dir, eval_meta, step=None, prefix=f"meta_{jax.host_id()}_")

  if eval_meta.bpd_round_id < num_bpd_rounds - 1:
    begin_ckpt = eval_meta.ckpt_id
    begin_bpd_round = eval_meta.bpd_round_id + 1
    begin_sampling_round = 0

  elif eval_meta.sampling_round_id < num_sampling_rounds - 1:
    begin_ckpt = eval_meta.ckpt_id
    begin_bpd_round = num_bpd_rounds
    begin_sampling_round = eval_meta.sampling_round_id + 1

  else:
    begin_ckpt = eval_meta.ckpt_id + 1
    begin_bpd_round = 0
    begin_sampling_round = 0

  rng = eval_meta.rng

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = imsize >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):

    # Restore and replicate the training state for executing on multiple devices
    state = restore_checkpoints(checkpoint_dir, state, ckpt)
    pstate = flax.jax_utils.replicate(state)


    if config.eval.enable_loss:
      losses_ = compute_loss_fn(eval_ds, scaler, rng, pstate, p_eval_step)
      save_loss(losses_, eval_dir, ckpt)

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      begin_repeat_id = begin_bpd_round // len(ds_bpd)
      begin_batch_id = begin_bpd_round % len(ds_bpd)
      # Repeat multiple times to reduce variance when needed
      for repeat in range(begin_repeat_id, bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for _ in range(begin_batch_id):
          next(bpd_iter)
        for batch_id in range(begin_batch_id, len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)
          rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
          bpd = likelihood_fn(jnp.asarray(step_rng), pstate, eval_batch['image'])[0].reshape(-1)
          bpds.extend(bpd)
          mean = jnp.mean(jnp.asarray(bpds))
          logging.info("ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, mean))
          bpd_round_id = batch_id + len(ds_bpd) * repeat

          # Save bits/dim to disk or Google Cloud Storage
          path = os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz")
          with tf.io.gfile.GFile(path, "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())

          # Save intermediate states to resume evaluation after pre-emption
          eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=bpd_round_id, rng=rng)
          save_checkpoint(eval_meta, step=ckpt * (num_sampling_rounds + num_bpd_rounds) + bpd_round_id)
    else:
      # Skip likelihood computation and save intermediate states for pre-emption
      eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=num_bpd_rounds - 1)
      save_checkpoint(eval_meta, step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_bpd_rounds - 1)

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      n_grid_samples = min(config.eval.n_grid_samples, config.eval.batch_size)
      state = jax.device_put(state)

      # Designed to be pre-emption safe. Automatically resumes when interrupted
      n = total_time = 0
      for r in range(begin_sampling_round, num_sampling_rounds):
        if jax.host_id() == 0:
          logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}_host_{jax.host_id()}")
        tf.io.gfile.makedirs(this_sample_dir)

        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
        start_time = time.time()
        samples, n_new = sampling_fn(sample_rng, pstate)
        total_time = total_time + (time.time() - start_time)
        n += n_new
        samples = samples.reshape((-1, imsize, imsize, 3))

        # Write samples to disk or Google Cloud Storage
        path = os.path.join(this_sample_dir, f"samples_{r}.npz")
        with tf.io.gfile.GFile(path, "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())
        
        # Showing a grid of samples
        path = os.path.join(this_sample_dir, f"sample{r}.png")
        with tf.io.gfile.GFile(path, "wb") as fout:
          image_grid = samples[0:n_grid_samples]
          nrow = int(np.sqrt(image_grid.shape[0]))
          utils.save_image(image_grid, fout, nrow=nrow, padding=2)
        
        # Force garbage collection before calling TensorFlow code for Inception network 
        # and again before returning to JAX code
        samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model, 1, inceptionv3)
        gc.collect()

        # Save latent represents of the Inception network to disk or Google Cloud Storage
        path = os.path.join(this_sample_dir, f"statistics_{r}.npz")
        with tf.io.gfile.GFile(path, "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
          fout.write(io_buffer.getvalue())

        # Update the intermediate evaluation state
        eval_meta = eval_meta.replace(ckpt_id=ckpt, sampling_round_id=r, rng=rng)
        # Save an intermediate checkpoint directly if not the last round.
        # Otherwise save eval_meta after computing the Inception scores and FIDs
        if r < num_sampling_rounds - 1:
          save_checkpoint(eval_meta, step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds)

      # Save eval_meta after computing IS/KID/FID to mark the end of evaluation for this checkpoint
      compute_image_scores(eval_dir, ckpt, config, num_sampling_rounds, inceptionv3, n, total_time)
      save_checkpoint(eval_meta, step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds)
      
    else:
      # Skip sampling and save intermediate evaluation states for pre-emption
      eval_meta = eval_meta.replace(ckpt_id=ckpt, sampling_round_id=num_sampling_rounds - 1, rng=rng)
      save_checkpoint(eval_meta, step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_sampling_rounds - 1 + num_bpd_rounds)

    begin_bpd_round = begin_sampling_round = 0

  cleanup_metafiles(eval_dir)


# Create FID stats by looping through the whole data
def fid_stats(config,
             fid_dir="assets/stats"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    fid_dir: The subfolder for storing fid statistics. 
  """
  # Create directory to eval_folder
  tf.io.gfile.makedirs(fid_dir)

  # Build data pipeline
  train_ds, eval_ds, dataset_builder = datasets.get_dataset(config,
                                              additional_dim=None,
                                              uniform_dequantization=False,
                                              evaluation=True,
                                              drop_remainder=False)
  bpd_iter = iter(train_ds)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  all_pools = []

  ## Alexia: We had to change to a while loop, because FFHQ doesn't have a dataset length!
  batch_id = -1
  while (True):
    try:
      batch = next(bpd_iter)
      batch_id = batch_id + 1
    except:
      break

    if jax.host_id() == 0:
      logging.info("Making FID stats -- step: %d" % (batch_id))

    batch_ = jax.tree_map(lambda x: x._numpy(), batch)
    batch_ = (batch_['image']*255).astype(np.uint8).reshape((-1, config.data.image_size, config.data.image_size, 3))

    # Force garbage collection before calling TensorFlow code for Inception network
    gc.collect()
    latents = evaluation.run_inception_distributed(batch_, inception_model,
                                                   inceptionv3=inceptionv3)
    all_pools.append(latents["pool_3"])
    # Force garbage collection again before returning to JAX code
    gc.collect()

  all_pools = np.concatenate(all_pools, axis=0) # Combine into one

  # Save latent represents of the Inception network to disk or Google Cloud Storage
  filename = f'{config.data.dataset.lower()}_{config.data.image_size}_stats.npz'
  with tf.io.gfile.GFile(
      os.path.join(fid_dir, filename), "wb") as fout:
    io_buffer = io.BytesIO()
    np.savez_compressed(
      io_buffer, pool_3=all_pools)
    fout.write(io_buffer.getvalue())


def setup_p_eval_step(config, sde, score_model):
    """ Creates the one-step evaluation function when loss computation is enabled. """
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, score_model,
                                   train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous, likelihood_weighting=likelihood_weighting)
    # Pmap (and jit-compile) multiple evaluation steps together for faster execution
    return jax.pmap(functools.partial(jax.lax.scan, eval_step), axis_name='batch', donate_argnums=1)


def create_likelihood_dataloaders(config):
  """ Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data. """
  kw = {'config': config, 'additional_dim': None, 'evaluation': True, 'uniform_dequantization': True}
  if config.eval.bpd_dataset.lower() == 'train':
    return datasets.get_dataset(**kw)[0], 1
  elif config.eval.bpd_dataset.lower() == 'test':
    return datasets.get_dataset(**kw)[1], 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")


def restore_checkpoints(checkpoint_dir, state, ckpt):
  # Wait if the target checkpoint doesn't exist yet
  PRINTED = False
  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
  while not tf.io.gfile.exists(ckpt_filename):
    if not PRINTED and jax.host_id() == 0:
      logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
      PRINTED = True
    time.sleep(60)

  # Wait for 2 additional mins in case the file exists but is not ready for reading
  try:
    state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
  except:
    time.sleep(60)
    try:
      state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
    except:
      time.sleep(120)
      state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
  return state


def compute_loss_fn(eval_ds, scaler, rng, pstate, p_eval_step):
  """ Computes the loss function on the full evaluation dataset if loss computation is enabled. """
  all_losses = []
  for i, batch in enumerate(iter(eval_ds)):  # pytype: disable=wrong-arg-types
    eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)  # pylint: disable=protected-access
    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
    next_rng = jnp.asarray(next_rng)
    (_, _), p_eval_loss = p_eval_step((next_rng, pstate), eval_batch)
    eval_loss = flax.jax_utils.unreplicate(p_eval_loss)
    all_losses.extend(eval_loss)
    if (i + 1) % 1000 == 0 and jax.host_id() == 0:
      logging.info("Finished %dth step loss evaluation" % (i + 1))


def save_loss(all_losses, eval_dir, ckpt):
  """ Saves loss values to disk or Google Cloud Storage. """
  all_losses = jnp.asarray(all_losses)
  with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
    io_buffer = io.BytesIO()
    np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
    fout.write(io_buffer.getvalue())


def compute_image_scores(eval_dir, ckpt, config, num_sampling_rounds, inceptionv3, n, total_time):
  """ Computes inception scores, FIDs and KIDs. """
  if jax.host_id() == 0:
    # Load all statistics that have been previously computed and saved for each host
    all_logits, all_pools = [], []
    for host in range(jax.host_count()):

      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}_host_{host}")
      path = os.path.join(this_sample_dir, "statistics_*.npz")
      stats = tf.io.gfile.glob(path)
      
      wait_message = False
      while len(stats) < num_sampling_rounds:
        if not wait_message:
          logging.warning("Waiting for statistics on host %d" % (host,))
          wait_message = True
        stats = tf.io.gfile.glob(path)
        time.sleep(30)

      for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
          stat = np.load(fin)
          if not inceptionv3:
            all_logits.append(stat["logits"])
          all_pools.append(stat["pool_3"])

    if not inceptionv3:
      all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
    all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(config)
    data_pools = data_stats["pool_3"]

    # Compute FID/KID/IS on all samples together.
    if not inceptionv3:
      inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
    else:
      inception_score = -1

    fid = tfgan.eval.frechet_classifier_distance_from_activations(
      data_pools, all_pools)
    # Hack to get tfgan KID work for eager execution.
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools)
    kid = tfgan.eval.kernel_classifier_distance_from_activations(
      tf_data_pools, tf_all_pools).numpy()
    del tf_data_pools, tf_all_pools

    if config.sampling.method.lower() == 'pc' and config.sampling.adaptive:
      logging.info(f"Method=LambaEM, h_init={config.sampling.h_init}, atol={config.sampling.abstol}, " 
                   f"rtol={config.sampling.reltol}, error_max={config.sampling.error_max}, "
                   f"error_use_prev={config.sampling.error_use_prev}, norm={config.sampling.norm}, "
                   f"error_per_unit_step={config.sampling.error_per_unit_step}, h_scale={config.sampling.h_scale}, "
                   f"safety={config.sampling.safety}, extrapolation={config.sampling.extrapolation}, "
                   f"sde_improved_euler={config.sampling.sde_improved_euler}, exp={config.sampling.exp}, "
                   f"order={config.sampling.order}, norm_treshold={config.sampling.norm_treshold}, "
                   f"sigma_scale={config.sampling.sigma_scale}, sde_discrete={config.sampling.sde_discrete}, "
                   f"correct_tweedie={config.sampling.correct_tweedie}, error_use_both={config.sampling.error_use_both}")
    else:
      logging.info(f"Method={config.sampling.method}, predictor={config.sampling.predictor}, corrector={config.sampling.corrector}")
    logging.info(
      "ckpt-%d --- n: %d, inception_score: %.6e, FID: %.6e, KID: %.6e, n_score_eval: %d, sampling_time (s): %d" % (
        ckpt, config.eval.num_samples, inception_score, fid, kid, n, total_time))

    with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"), "wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
      f.write(io_buffer.getvalue())
  else:
    # For host_id() != 0.
    # Use file existence to emulate synchronization across hosts
    while not tf.io.gfile.exists(os.path.join(eval_dir, f"report_{ckpt}.npz")):
      time.sleep(1.)


def cleanup_metafiles(eval_dir):
  """ Removes all meta files after finishing evaluation. """
  meta_path = os.path.join(eval_dir, f"meta_{jax.host_id()}_*")
  for file in tf.io.gfile.glob(meta_path):
    tf.io.gfile.remove(file)

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
# pytype: skip-file
"""Various sampling methods."""
import functools

import jax
import jax.numpy as jnp
import jax.random as random
import abc
import flax

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from utils import batch_mul, batch_mul_3, batch_add

from models import utils as mutils
#from diffeqpy import de
from jax.experimental.host_callback import id_print, barrier_wait

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, model, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  cs = config.sampling
  sampler_name = cs.method
  args = {'sde': sde, 'model': model, 'shape': shape, 'inverse_scaler': inverse_scaler, 'eps': eps}

  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    return get_ode_sampler(denoise=cs.noise_removal, **args)

  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(cs.predictor.lower())
    corrector = get_corrector(cs.corrector.lower())

    kwargs = {key: getattr(cs, key) for key in get_pc_sampler.__code__.co_varnames if hasattr(cs, key)}
    addargs = {'predictor': predictor, 
               'predictor_name': cs.predictor.lower(),
               'corrector': corrector,
               'corrector_name': cs.corrector.lower(),
               'denoise': cs.noise_removal, 
               'n_steps': cs.n_steps_each if cs.corrector.lower() != "none" else 0}
    return get_pc_sampler(**{**kwargs, **args, **addargs})

  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, shape=None, probability_flow=False, eps=1e-3, abstol = 1e-2, reltol = 1e-2, 
    error_use_prev=True, norm = "L2_scaled", safety = .9, sde_improved_euler=True, extrapolation = True, exp=0.9):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, rng, x, t, h, x_prev):
    """One update of the predictor.

    Args:
      rng: A JAX random state.
      x: A JAX array representing the current state
      t: A JAX array representing the current time step.
      h: scalar: step-size taken

    Returns:
      x: A JAX array of the next state.
      x_mean: A JAX array. The next state without random noise. Useful for denoising.
    """
    pass

# Previous version was incorrect since it was taking step 1/1000=1e-3, but t is not always in increment of 1e-3 due to the linspace stopping at eps
@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, shape=None, probability_flow=False, eps=1e-3, abstol = 1e-2, reltol = 1e-2, 
    error_use_prev=True, norm = "L2_scaled", safety = .9, sde_improved_euler=True, extrapolation = True, exp=0.9):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, rng, x, t, h, x_prev=None):
    my_sde = self.rsde.sde

    z = random.normal(rng, x.shape)
    drift, diffusion = my_sde(x, t)
    x_mean = x - drift * h
    x = x_mean + batch_mul(diffusion, jnp.sqrt(h) * z)
    return x, x_mean

# EM or Improved-Euler (Heun's method) with adaptive step-sizes
@register_predictor(name='adaptive')
class AdaptivePredictor(Predictor):
  def __init__(self, sde, score_fn, shape, probability_flow=False, eps=1e-3, abstol = 1e-2, reltol = 1e-2,
    error_use_prev=True, norm = "L2_scaled", safety = .9, sde_improved_euler=True, extrapolation = True, exp=0.9):
    super().__init__(sde, score_fn, probability_flow)
    self.h_min = 1e-10 # min step-size
    self.t = sde.T # starting t
    self.eps = eps # end t
    self.abstol = abstol
    self.reltol = reltol
    self.error_use_prev = error_use_prev
    self.norm = norm
    self.safety = safety
    self.sde_improved_euler = sde_improved_euler
    self.extrapolation = extrapolation
    self.n = shape[1]*shape[2]*shape[3] #size of each sample
    self.exp = exp
    
    if self.norm == "L2_scaled":
      def norm_fn(x):
        return jnp.sqrt(jnp.sum((x)**2, axis=(1,2,3), keepdims=True)/self.n)
    elif self.norm == "L2":
      def norm_fn(x):
        return jnp.sqrt(jnp.sum((x)**2, axis=(1,2,3), keepdims=True))
    elif self.norm == "Linf":
      def norm_fn(x):
        return jnp.max(jnp.abs(x), axis=(1,2,3), keepdims=True)
    else:
      raise NotImplementedError(self.norm)
    self.norm_fn = norm_fn


  def update_fn(self, rng, x, t, h, x_prev): 
    # Note: both h and t are vectors with batch_size elems (this is because we want adaptive step-sizes for each sample separately)
    my_rsde = self.rsde.sde

    h_ = jnp.expand_dims(h, (1,2,3)) # expand for multiplications
    t_ = jnp.expand_dims(t, (1,2,3)) # expand for multiplications

    z = random.normal(rng, x.shape)
    drift, diffusion = my_rsde(x, t)

    if not self.sde_improved_euler: # Like Lamba's algorithm
      x_mean_new = x - batch_mul(h_, drift)
      drift_Heun, _ = my_rsde(x_mean_new, t - h) # Heun's method on the ODE
      if self.extrapolation: # Extrapolate using the Heun's method result
        x_mean_new = x - batch_mul(h_/2, drift + drift_Heun)
      x_new = x_mean_new + batch_mul_3(diffusion, jnp.sqrt(h_), z)
      E = batch_mul(h_/2, drift_Heun - drift) # local-error between EM and Heun (ODEs)
      x_check = x_mean_new
    else:
      # Heun's method for SDE (while Lamba method only focuses on the non-stochastic part, this also includes the stochastic part)
      K1_mean = -batch_mul(h_, drift)
      K1 = K1_mean + batch_mul_3(diffusion, jnp.sqrt(h_), z)

      drift_Heun, diffusion_Heun = my_rsde(x + K1, t - h)
      K2_mean = -batch_mul(h_, drift_Heun)
      K2 = K2_mean + batch_mul_3(diffusion_Heun, jnp.sqrt(h_), z)
      E = 1/2*(K2 - K1) # local-error between EM and Heun (SDEs) (right one)
      #E = 1/2*(K2_mean - K1_mean) # a little bit better with VE, but not that much
      if self.extrapolation: # Extrapolate using the Heun's method result
        x_new = x + (1/2)*(K1 + K2)
        x_check = x + K1
        x_check_other = x_new
      else:
        x_new = x + K1
        x_check = x + (1/2)*(K1 + K2)
        x_check_other = x_new

    # Calculating the error-control
    if self.error_use_prev:
      reltol_ctl = jnp.maximum(jnp.abs(x_prev), jnp.abs(x_check))*self.reltol
    else:
      reltol_ctl = jnp.abs(x_check)*self.reltol
    err_ctl = jnp.maximum(reltol_ctl, self.abstol)

    # Normalizing for each sample separately
    E_scaled_norm = self.norm_fn(E/err_ctl)

    # Accept or reject x_{n+1} and t_{n+1} for each sample separately
    accept = jax.vmap(lambda a: a <= 1)(E_scaled_norm)
    x = jnp.where(accept, x_new, x)
    x_prev = jnp.where(accept, x_check, x_prev)
    t_ = jnp.where(accept, t_ - h_, t_)

    # Change the step-size
    h_max = jnp.maximum(t_ - self.eps, 0) # max step-size must be the distance to the end (we use maximum between that and zero in case of a tiny but negative value: -1e-10)
    E_pow = jnp.where(h_ == 0, h_, jnp.power(E_scaled_norm, -self.exp))  # Only applies power when not zero, otherwise, we get nans
    h_new = jnp.minimum(h_max, self.safety*h_*E_pow)

    return x, x_prev, t_.reshape((-1)), h_new.reshape((-1))


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, shape=None, probability_flow=False, eps=1e-3, abstol = 1e-2, reltol = 1e-2, 
    error_use_prev=True, norm = "L2_scaled", safety = .9, sde_improved_euler=True, extrapolation = True, exp=0.9):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, rng, x, t, h=None, x_prev=None):
    f, G = self.rsde.discretize(x, t)
    z = random.normal(rng, x.shape)
    x_mean = x - f
    x = x_mean + batch_mul(G, z)
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, shape=None, probability_flow=False, eps=1e-3, abstol = 1e-2, reltol = 1e-2, 
    error_use_prev=True, norm = "L2_scaled", safety = .9, sde_improved_euler=True, extrapolation = True, exp=0.9):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, rng, x, t, h=None, x_prev=None):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = jnp.where(timestep == 0, jnp.zeros(t.shape), sde.discrete_sigmas[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + batch_mul(score, sigma ** 2 - adjacent_sigma ** 2)
    std = jnp.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, noise)
    return x, x_mean

  def vpsde_update_fn(self, rng, x, t, h=None, x_prev=None):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    beta = sde.discrete_betas[timestep]
    score = self.score_fn(x, t)
    x_mean = batch_mul((x + batch_mul(beta, score)), 1. / jnp.sqrt(1. - beta))
    noise = random.normal(rng, x.shape)
    x = x_mean + batch_mul(jnp.sqrt(beta), noise)
    return x, x_mean

  def update_fn(self, rng, x, t, h=None, x_prev=None):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(rng, x, t, h, x_prev)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(rng, x, t, h, x_prev)

@register_predictor(name='ddim')
class DDIMPredictor(Predictor):
  """Based on https://arxiv.org/pdf/2010.02502.pdf, version with no noise, only support VP process"""

  def __init__(self, sde, score_fn, shape=None, probability_flow=False, eps=1e-3, abstol = 1e-2, reltol = 1e-2, 
    error_use_prev=True, norm = "L2_scaled", safety = .9, sde_improved_euler=True, extrapolation = True, exp=0.9):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vpsde_update_fn(self, rng, x, t, h, x_prev=None):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    timestep_next = ((t-h) * (sde.N - 1) / sde.T).astype(jnp.int32) # same exact thing as  timestep - 1
    alpha = sde.alphas_cumprod[timestep]
    alpha_next = sde.alphas_cumprod[timestep_next]
    score = -batch_mul(self.score_fn(x, t), jnp.sqrt(1-alpha)) # From Yang score-function to Ho "score-function"
    x = batch_mul(jnp.sqrt(alpha_next),batch_mul(x - batch_mul(jnp.sqrt(1. - alpha), score), 1. / jnp.sqrt(alpha))) + batch_mul(jnp.sqrt(1-alpha_next), score)
    return x, x

  def update_fn(self, rng, x, t, h, x_prev=None):
    return self.vpsde_update_fn(rng, x, t, h, x_prev)

@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, shape=None, probability_flow=False, eps=1e-3, abstol = 1e-2, reltol = 1e-2, 
    error_use_prev=True, norm = "L2_scaled", safety = .9, sde_improved_euler=True, extrapolation = True, exp=0.9):
    pass

  def update_fn(self, rng, x, t, h=None, x_prev=None):
    return x, x


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, rng, x, t):
    """One update of the corrector.

    Args:
      rng: A JAX random state.
      x: A JAX array representing the current state
      t: A JAX array representing the current time step.

    Returns:
      x: A JAX array of the next state.
      x_mean: A JAX array. The next state without random noise. Useful for denoising.
    """
    pass


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, rng, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      alpha = sde.alphas[timestep]
    else:
      alpha = jnp.ones_like(t)

    def loop_body(step, val):
      rng, x, x_mean = val
      grad = score_fn(x, t)
      rng, step_rng = jax.random.split(rng)
      noise = jax.random.normal(step_rng, x.shape)
      grad_norm = jnp.linalg.norm(
        grad.reshape((grad.shape[0], -1)), axis=-1).mean()
      grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')
      noise_norm = jnp.linalg.norm(
        noise.reshape((noise.shape[0], -1)), axis=-1).mean()
      noise_norm = jax.lax.pmean(noise_norm, axis_name='batch')
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + batch_mul(step_size, grad)
      x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, rng, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      alpha = sde.alphas[timestep]
    else:
      alpha = jnp.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    def loop_body(step, val):
      rng, x, x_mean = val
      grad = score_fn(x, t)
      rng, step_rng = jax.random.split(rng)
      noise = jax.random.normal(step_rng, x.shape)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + batch_mul(step_size, grad)
      x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, rng, x, t):
    return x, x


def shared_predictor_update_fn(rng, state, x, t, h=None, x_prev=None, sde=None, shape=None, model=None, predictor=None, probability_flow=None, continuous=None, 
  eps=1e-3, abstol = 1e-2, reltol = 1e-2, error_use_prev=True, norm = "L2_scaled", safety = .9, extrapolation = False, sde_improved_euler=False, exp = 0.9):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, shape, probability_flow, eps=eps, abstol=abstol, reltol=reltol, 
      error_use_prev=error_use_prev, norm = norm, safety = safety, extrapolation = extrapolation, sde_improved_euler=sde_improved_euler, exp = exp)
  return predictor_obj.update_fn(rng, x, t, h, x_prev)


def shared_corrector_update_fn(rng, state, x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(rng, x, t)


def get_pc_sampler(sde, model, shape, predictor, predictor_name, corrector, corrector_name, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True,
                   eps=1e-3, h_init=1e-2, abstol = 1e-2, reltol = 1e-2, 
                   error_use_prev=True, norm = "L2_scaled", safety = .9, 
                   extrapolation = True, sde_improved_euler=True, exp = 0.9):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.

  Returns:
    A sampling function that takes random states, and a replcated training state and returns samples.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          shape=shape,
                                          model=model,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous, 
                                          eps=eps, abstol = abstol, reltol = reltol, 
                                          error_use_prev=error_use_prev, norm = norm, safety = safety, 
                                          extrapolation = extrapolation, sde_improved_euler=sde_improved_euler, 
                                          exp=exp)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          model=model,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)
  def pc_sampler(rng, state):
    """ The PC sampler funciton.

    Args:
      rng: A JAX random state
      state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
    Returns:
      Samples
    """
    # Initial sample
    rng, step_rng = random.split(rng)
    x = sde.prior_sampling(step_rng, shape)
    timesteps = jnp.linspace(sde.T, eps, sde.N)
    h = timesteps - jnp.append(timesteps, 0)[1:] # true step-size: difference between current time and next time (only the new predictor classes will use h, others will ignore)
    N = sde.N - 1

    def loop_body(i, val):
      rng, x, x_mean = val
      t = timesteps[i]
      vec_t = jnp.ones(shape[0]) * t
      rng, step_rng = random.split(rng)
      x, x_mean = corrector_update_fn(step_rng, state, x, vec_t)
      rng, step_rng = random.split(rng)
      x, x_mean = predictor_update_fn(step_rng, state, x, vec_t, h[i])
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, N, loop_body, (rng, x, x))
    if denoise: # Tweedie formula
      eps_t = jnp.ones(shape[0]) * eps
      score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous)
      u, std = sde.marginal_prob(x, eps_t)
      x = x + batch_mul(std ** 2, score_fn(x, eps_t))
    return inverse_scaler(x), N * (n_steps + 1) + 1

  def pc_sampler_adaptive(rng, state):
    """ The PC sampler funciton.

    Args:
      rng: A JAX random state
      state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
    Returns:
      Samples
    """
    # Initial sample
    rng, step_rng = random.split(rng)
    x = sde.prior_sampling(step_rng, shape)
    h = jnp.ones(shape[0]) * h_init # initial step_size
    t = jnp.ones(shape[0]) * sde.T # initial time

    def loop_body(val):
      rng, x, x_prev, t, h, i = val
      rng, step_rng = random.split(rng)
      x, x_prev = corrector_update_fn(step_rng, state, x, t)
      if corrector_name  != "none":
        i = i + 1
      rng, step_rng = random.split(rng)
      x, x_prev, t, h = predictor_update_fn(step_rng, state, x, t, h, x_prev)
      if predictor_name  != "none":
        i = i + 2
      return rng, x, x_prev, t, h, i

    def condition_continue(val):
      rng, x, x_prev, t, h, i = val
      return (jnp.abs(t - eps) > 1e-6).any()

    _, x, _, _, _, n_iter = jax.lax.while_loop(condition_continue, loop_body, (rng, x, x, t, h, 0))

    if denoise: # Tweedie formula
      eps_t = jnp.ones(shape[0]) * eps
      score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous)
      u, std = sde.marginal_prob(x, eps_t)
      x = x + batch_mul(std ** 2, score_fn(x, eps_t))
    return inverse_scaler(x), n_iter

  def my_sampling(rng, state):
    if predictor_name == "adaptive":
      results, n = jax.pmap(pc_sampler_adaptive, axis_name='batch')(rng, state)
    else:
      results, n = jax.pmap(pc_sampler, axis_name='batch')(rng, state)
    return results, flax.jax_utils.unreplicate(n)

  return my_sampling


def get_ode_sampler(sde, model, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.

  Returns:
    A sampling function that takes random states, and a replicated training state and returns samples.
  """

  @jax.pmap
  def denoise_update_fn(state, x): # Tweedie formula
    eps_t = jnp.ones(shape[0]) * eps
    score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
    u, std = sde.marginal_prob(x, eps_t)
    return x + batch_mul(std ** 2, score_fn(x, eps_t))

  @jax.pmap
  def drift_fn(state, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(prng, pstate, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      prng: An array of random state. The leading dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      z: If present, generate samples from latent code `z`.
    """
    # Initial sample
    rng = flax.jax_utils.unreplicate(prng)
    rng, step_rng = random.split(rng)
    if z is None:
      # If not represent, sample the latent code from the prior distibution of the SDE.
      x = sde.prior_sampling(step_rng, (jax.local_device_count(),) + shape)
    else:
      x = z

    def ode_func(t, x):
      x = from_flattened_numpy(x, (jax.local_device_count(),) + shape)
      vec_t = jnp.ones((x.shape[0], x.shape[1])) * t
      drift = drift_fn(pstate, x, vec_t)
      return to_flattened_numpy(drift)

    # Black-box ODE solver for the probability flow ODE
    solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                   rtol=rtol, atol=atol, method=method)
    nfe = solution.nfev
    x = jnp.asarray(solution.y[:, -1]).reshape((jax.local_device_count(),) + shape)

    if denoise:
      x = denoise_update_fn(pstate, x)
      nfe = nfe + 1

    x = inverse_scaler(x)
    return x, nfe

  return ode_sampler


def to_flattened(x):
  """Flatten a JAX array `x` and convert it to numpy."""
  return x.reshape((-1,)).tolist()

def from_flattened(x, shape):
  """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
  return jnp.asarray(x).reshape(shape)

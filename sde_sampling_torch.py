## Modified from https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py
## Will only work if https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py is in the same folder
## I removed all the stuff we do not need

#### Defaults
# config.sampling.method='adaptive', choices=['euler_maruyama','adaptive']
# config.sampling.sampling_h_init=1e-2
# config.sampling.sampling_reltol=1e-2
# config.sampling.sampling_safety=0.9
# config.sampling.sampling_exp=0.9

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
import torch
import abc
import sde_lib
import numpy as np
from datetime import datetime

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


def get_predictor(name):
  return _PREDICTORS[name]


def get_sampling_fn(config, sde, shape, eps, device):
  """Create a sampling function.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  predictor = get_predictor(config.sampling.method)

  sampling_fn = get_pc_sampler(sde = sde,
                               shape = shape,
                               predictor = predictor,
                               denoise = True,
                               eps = eps,
                               device = device,
                               abstol = config.sampling.sampling_abstol, 
                               reltol = config.sampling.sampling_reltol, 
                               safety = config.sampling.sampling_safety, 
                               exp = config.sampling.sampling_exp,
                               adaptive = config.sampling.method == "adaptive",
                               h_init = config.sampling.sampling_h_init)

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, shape=None, eps=1e-3, 
    abstol = 1e-2, reltol = 1e-2, safety = .9, exp=0.9):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t, h, x_prev):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, shape=None, eps=1e-3, 
    abstol = 1e-2, reltol = 1e-2, safety = .9, exp=0.9):
    super().__init__(sde, score_fn)

  def update_fn(self, x, t, h, x_prev=None):

    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x - drift * h
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(h) * z
    return x, x_mean

# EM or Improved-Euler (Heun's method) with adaptive step-sizes
@register_predictor(name='adaptive')
class AdaptivePredictor(Predictor):
  def __init__(self, sde, score_fn, shape, eps=1e-3, 
    abstol = 1e-2, reltol = 1e-2, safety = .9, exp=0.9):
    super().__init__(sde, score_fn)
    self.h_min = 1e-10 # min step-size
    self.t = sde.T # starting t
    self.eps = eps # end t
    self.abstol = abstol
    self.reltol = reltol
    self.error_use_prev = True
    self.safety = safety
    self.n = shape[1]*shape[2]*shape[3] #size of each sample
    self.exp = exp
    
    #"L2_scaled":
    def norm_fn(x):
      return torch.sqrt(torch.sum((x)**2, dim=(1,2,3), keepdim=True)/self.n)
    self.norm_fn = norm_fn


  def update_fn(self, x, t, h, x_prev): 
    # Note: both h and t are vectors with batch_size elems (this is because we want adaptive step-sizes for each sample separately)
    my_rsde = self.rsde.sde

    h_ = h[:, None, None, None] # expand for multiplications
    t_ = t[:, None, None, None] # expand for multiplications
    z = torch.randn_like(x)
    drift, diffusion = my_rsde(x, t)

    # Heun's method for SDE (while Lamba method only focuses on the non-stochastic part, this also includes the stochastic part)
    K1_mean = -h_ * drift
    K1 = K1_mean + diffusion[:, None, None, None] * torch.sqrt(h_) * z

    drift_Heun, diffusion_Heun = my_rsde(x + K1, t - h)
    K2_mean = -h_*drift_Heun
    K2 = K2_mean + diffusion_Heun[:, None, None, None] * torch.sqrt(h_) * z
    E = 1/2*(K2 - K1) # local-error between EM and Heun (SDEs) (right one)
    #E = 1/2*(K2_mean - K1_mean) # a little bit better with VE, but not that much
    # Extrapolate using the Heun's method result
    x_new = x + (1/2)*(K1 + K2)
    x_check = x + K1
    x_check_other = x_new

    # Calculating the error-control
    if self.error_use_prev:
      reltol_ctl = torch.maximum(torch.abs(x_prev), torch.abs(x_check))*self.reltol
    else:
      reltol_ctl = torch.abs(x_check)*self.reltol
    err_ctl = torch.clamp(reltol_ctl, min=self.abstol)

    # Normalizing for each sample separately
    E_scaled_norm = self.norm_fn(E/err_ctl)

    # Accept or reject x_{n+1} and t_{n+1} for each sample separately
    accept = E_scaled_norm <= torch.ones_like(E_scaled_norm)
    x = torch.where(accept, x_new, x)
    x_prev = torch.where(accept, x_check, x_prev)
    t_ = torch.where(accept, t_ - h_, t_)

    # Change the step-size
    h_max = torch.clamp(t_ - self.eps, min=0) # max step-size must be the distance to the end (we use maximum between that and zero in case of a tiny but negative value: -1e-10)
    E_pow = torch.where(h_ == 0, h_, torch.pow(E_scaled_norm, -self.exp))  # Only applies power when not zero, otherwise, we get nans
    h_new = torch.minimum(h_max, self.safety*h_*E_pow)

    return x, x_prev, t_.reshape((-1)), h_new.reshape((-1))

def shared_predictor_update_fn(x, t, h, sde, model, predictor, x_prev=None, shape=None,
    eps=1e-3, abstol = 1e-2, reltol = 1e-2, safety = .9, exp=0.9):
  """A wrapper that configures and returns the update function of predictors."""
  predictor_obj = predictor(sde, model, shape=shape, eps=eps, 
    abstol = abstol, reltol = reltol, safety = safety, exp=0.9)
  return predictor_obj.update_fn(x, t, h, x_prev)


def get_pc_sampler(sde, shape, predictor,
                   denoise=True, device='cuda',
                   eps=1e-3, abstol = 1e-2, reltol = 1e-2, safety = .9, exp=0.9, adaptive=False, h_init = 1e-2):
  """Create a SDE sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          shape=shape,
                                          predictor=predictor,
                                          eps=eps, abstol = abstol, reltol = reltol, 
                                          safety = safety, 
                                          exp=exp)

  def pc_sampler(model, prior=None):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      #print(datetime.now().time())
      # Initial sample
      if prior is None:
        x = sde.prior_sampling(shape).to(device)
      else:
        x = prior.to(device)
      timesteps = np.linspace(sde.T, eps, sde.N)
      h = timesteps - np.append(timesteps, 0)[1:] # true step-size: difference between current time and next time (only the new predictor classes will use h, others will ignore)
      N = sde.N - 1

      for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0]).to(device) * t
        x, x_mean = predictor_update_fn(x, vec_t, h[i], model=model)


      if denoise:
        eps_t = torch.ones(shape[0]).to(device) * eps
        u, std = sde.marginal_prob(x, eps_t)
        x = x + (std[:, None, None, None] ** 2) * model(x, eps_t)
      #print(datetime.now().time())
      return x, N + 1

  def pc_sampler_adaptive(model, prior=None):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      #print(datetime.now().time())
      # Initial sample
      if prior is None:
        x = sde.prior_sampling(shape).to(device)
      else:
        x = prior.to(device)
      h = torch.ones(shape[0]).to(device) * h_init # initial step_size
      t = torch.ones(shape[0]).to(device) * sde.T # initial time
      x_prev = x 

      N = 0
      while (torch.abs(t - eps) > 1e-6).any():
        x, x_prev, t, h = predictor_update_fn(x, t, h, x_prev=x_prev, model=model)
        N = N + 1

      if denoise:
        eps_t = torch.ones(shape[0]).to(device) * eps
        u, std = sde.marginal_prob(x, eps_t)
        x = x + (std[:, None, None, None] ** 2) * model(x, eps_t)
      #print(datetime.now().time())
      return x, N + 1

  if adaptive:
    return pc_sampler_adaptive
  else:
    return pc_sampler


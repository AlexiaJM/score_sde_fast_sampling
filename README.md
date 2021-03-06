# Gotta Go Fast When Generating Data with Score-Based Models

This repo contains the official implementation for the paper [Gotta Go Fast When Generating Data with Score-Based Models](https://arxiv.org/abs/2105.14080), which shows how to generate data as fast as possible with score-based models using a well-designed SDE solver. See the [blog post](https://ajolicoeur.wordpress.com/gotta-go-fast-with-score-based-models/) for more details.

<p align="center">
  <img src="https://i.kym-cdn.com/photos/images/newsfeed/000/615/826/8ba.gif">
</p>

--------------------

This code is a heavy modification of the [Generative Modeling through Stochastic Differential Equations repository](https://github.com/yang-song/score_sde).

## 2021-09-21: Using the GottaGoFast sampler in Torch
If you want the PyTorch version, please use https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py along with my modified version of https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py which is here: https://github.com/AlexiaJM/score_sde_fast_sampling/blob/main/sde_sampling_torch.py.

Notes:
- I removed the other methods pc methods in this new file, feel free to add them back based on https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py
- I removed a lot of options which were unnecessary from the algorithm to simplify things

## To run the experiments in the paper

See the [requirements](https://github.com/AlexiaJM/score_sde_fast_sampling/blob/main/requirements.txt). 
Change the settings and folders in https://github.com/AlexiaJM/score_sde_fast_sampling/blob/main/experiments.sh and run parts of the script to run the CIFAR-10, LSUN-Church, and FFHQ experiments.

The SDE solver can be found [here](https://github.com/AlexiaJM/score_sde_fast_sampling/blob/main/sampling.py#L172) and the loop [here](https://github.com/AlexiaJM/score_sde_fast_sampling/blob/main/sampling.py#L574).

## For general usage 

Please refer to the [original code](https://github.com/yang-song/score_sde).

## Pretrained checkpoints

https://drive.google.com/drive/folders/10pQygNzF7hOOLwP3q8GiNxSnFRpArUxQ?usp=sharing

## References
If you find the code useful for your research, please consider citing
```bib
@article{jolicoeurmartineau2021gotta,
      title={Gotta Go Fast When Generating Data with Score-Based Models}, 
      author={Alexia Jolicoeur-Martineau and Ke Li and R{\'e}mi Pich{\'e}-Taillefer and Tal Kachman and Ioannis Mitliagkas},
      journal={arXiv preprint arXiv:2105.14080},
      year={2021}
}
```
and 
```bib
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```

Official theme song can be found here: https://soundcloud.com/emyaze/gotta-go-fast.

## Samples (see the paper for more samples)

![](/images/both.png)


# PackeRL v2, Placer, and LOGGIA

*PackeRL* is an RL training and evaluation framework for routing optimization in computer networks. 
It leverages the [ns-3 network simulator](https://www.nsnam.org) for packet-level simulation dynamics. 
PackeRL is designed to be modular and flexible, allowing for easy integration of new algorithms and network models.

Besides PackeRL version 2 (the first version is available via https://github.com/SAP-archive/packerl), this repository comes with our new policy design *LOGGIA* and *Placer*, published in [TMLR 09.2026](https://openreview.net/pdf?id=jmk2SoQklg) and [MaLeNe 2026](https://arxiv.org/abs/2602.12798) respectively (check the papers for further info).

### Documentation / Getting started

Extensive documentation is available in the [documentation](doc/README.md). Please note that *LOGGIA*, in this codebase, is the evolution of *M-Slim* from the [PackeRL project](https://alrhub.github.io/packerl-website/) and denoted as `ms` in the experiment configurations (whereas the original *M-Slim* is denoted as `ms_tmlr24`).

### Citation

This repository is part of the LOGGIA project ([webpage](https://alrhub.github.io/loggia-website/), [paper](https://openreview.net/pdf?id=jmk2SoQklg)) published in TMLR 2026. If you use this framework or *LOGGIA* in your research, please
cite the LOGGIA paper:

```
@article{
boltres2026towards,
title={Towards Near-Real-Time Telemetry-Aware Routing with Neural Routing Algorithms},
author={Boltres, Andreas and Freymuth, Niklas and Schichtholz, Benjamin and K{\"o}nig, Michael and Neumann, Gerhard},
journal={Transactions on Machine Learning Research},
year={2026},
url={https://openreview.net/forum?id=jmk2SoQklg}
}
```

Additionally, if you use *Placer* in your research, please cite the following paper:

```
@inproceedings{boltres2026can,
  title={Can Neural Networks Provide Latent Embeddings for Telemetry-Aware Greedy Routing?},
  author={Boltres, Andreas and Freymuth, Niklas and Neumann, Gerhard},
  booktitle={7th KuVS Fachgespr{\"a}ch on Machine Learning in Networking (MaLeNe)},
  year={2026},
  url={https://arxiv.org/abs/2602.12798}
}
```

### Development/Maintenance Notice

Please note that this is a research artifact and will thus not receive customer-facing maintenance or support.

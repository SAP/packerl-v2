# Getting started Quickly

Tested and used on Ubuntu 20.04/24.04 LTS, RHEL 8.4/9.4, and SUSE 15.5. All other setups at your own risk!

### Core Components of the Repo

```

config/             # configuration files for experiments, incl. presets and default configs
data/               # data consumed by experiments, e.g. scenario datasets or pre-trained models
doc/                # tutorial files and documentation
ns3/                # [git submodule] ns-3 network simulator
ns3-ai/             # [git submodule] ns-3-ai module for interacting with AI/ML algorithms
ns3-packerl/        # our ns-3 extension module for PackeRL
  - applications/   # custom ns-3 applications for PackeRL
  - monitoring/     # custom ns-3 monitoring modules for PackeRL
  - odd-routing/    # custom ns-3 routing module for PackeRL that can deal with our action space design
  - utils/          # utility functions and classes for ns-3 (e.g. shared memory implementation)
  packerl.cc/h      # C++ code for the ns-3 side of the PackeRL environment
  sim.cc            # C++ entry point for the ns-3 simulation (called from the python side)
scripts/            # convenience scripts, e.g. for debugging and plotting results
src/                # the principal python source directory
  - algorithm/      # contains all things related to routing algorithms
    - baselines/    # non-ML routing baselines
    - trainable/    # ML routing mechanisms
  - nn/             # neural network modules and architectures
  - normalizer/     # observation/reward normalizers for trainable modules
  - scenarios/      # scenario generation logic
  - trainer/        # training algorithms (Behavioral Cloning, Reinforcement Learning, ...)
  - utils/          # utility functions and classes  (e.g. shared memory implementation, observation graph logic)
  - packerl_env.py  # Python environment for PackeRL with Gymnasium-like interface for RL logic
test/               # unit and manual tests for the PackeRL codebase
install.sh          # PackeRL installation script for all components and dependencies
main.py             # The entry point for running PackeRL experiments
```

## Installation - Variant 1 (local)

### Requirements

- `bash` (or any other shell you're comfortable with)
- `git`
- `conda` (https://github.com/conda-forge/miniforge is recommended)

**BwHPC 3.0**

Put the following lines into your `.bashrc` to start with everything set for PackeRL experiments when logging into BwHPC.

```
module load devel/miniforge/24.11.0-python-3.12
# For some reason the ns3 build process uses the wrong compiler when trying to install ns3 on the cluster. 
# The following two lines make sure you're using a suitable compiler.
module load compiler/gnu/14.2
module load devel/cuda/11.8  # if you want to use GPU acceleration
export LD_PRELOAD=/opt/bwhpc/common/compiler/gnu/14.2.0/lib64/libstdc++.so.6
export CONDA_EXE="/opt/bwhpc/common/devel/miniforge/24.11.0-py3.12/bin/conda"

eval "$(conda shell.bash hook)"
```

### Downloading PackeRL

```
git clone <this repo> --recurse-submodules
```

if you cloned the repository without its submodules, go to the root of the repository and initialize the submodules:

```
cd packerl
git submodule update --init --recursive
```

### Installation commands (from repository root folder)

```
bash install.sh
conda activate packerl
```

Check the help of `install.sh` with `-h` for more options, e.g.:

- In order to enable the MAGNNETO baseline, use the `-m` flag.
- In order to enable GPU support (CUDA 11.8), add `-p cu118` to the install command. 
    We do not support other CUDA versions at the moment.


## Installation - Variant 2 (docker)
Info: Ubuntu 24.04 LTS used as base image .

### Requirements
- Docker

### Downloading PackeRL

```
git clone <this repo> --recurse-submodules
```

if you cloned the repository without its submodules, go to the root of the repository and initialize the submodules:

```
cd packerl
git submodule update --init --recursive
```

### Building the Docker container
```
#in root of the repository
docker compose build
```

### Starting the Docker container
```
#in root of the repository
docker compose up -d
```

### Entering the Docker container and running PackeRL
```
docker exec -it packerl bash
$ source entrypoint.sh
$ python main.py #see below
```

### Stopping the Docker container
```
#in root of the repository
docker compose stop
```

## Quick Run

In the root of the repository:

```
python main.py +_runs=sanity_check_single
```

The last part of this command selects the `sanity_check_single.yaml` preset file from the `config/_runs/` folder, 
which is a minimal setup for testing the installation. 

We use `hydra` for managing our experiment configurations. For further information on creating 
and running sets of experiments, see [Section 9](09_experimentation.md).



## Updating the repo

1. Run `git pull --recurse-submodules` to get the latest updates for the repo and its submodules.
2. Run `bash install.sh -r` to (re-)install everything, re-building modified components.

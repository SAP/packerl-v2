#!/bin/bash

# === Helper Functions ===

display_help() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  -e            Delete and re-build the conda environment even if it exists.
  -r            Rebuild ns3 (placeholder in this script).
  -d            Build in default mode with debug logs. If not, builds in optimized mode without logs.
  -m            Install the requirements for using the MAGNNETO baseline (i.e. tensorflow).
  -p <platform> Desired PyTorch platform: "cpu" or "cu118".
                The PyTorch stack is (re)installed only if:
                  * the environment is (re)created or missing, OR
                  * -p is given AND it differs from the currently installed platform.
  -h            Display this help message.

Examples:
  ./install.sh -e                 # recreate env; installs torch stack (defaults to cpu)
  ./install.sh -e -p cu118        # recreate env; installs CUDA 11.8 torch stack
  ./install.sh -p cu118           # switch existing env to CUDA 11.8 only if different
EOF
}

# Bootstrap conda so `conda activate` works in this shell
bootstrap_conda() {
  if base="$(conda info --base 2>/dev/null)" && [ -f "$base/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$base/etc/profile.d/conda.sh"
  elif [ -n "$CONDA_EXE" ]; then
    eval "$("$CONDA_EXE" shell.bash hook)"
  else
    echo "ERROR: Could not bootstrap Conda. Neither \`conda info --base\` worked nor \$CONDA_EXE is set."
    echo "Please set CONDA_EXE to the full path of your \`conda\` executable and re-run."
    return 1
  fi
}

# Detect currently installed torch platform: prints one of: cpu | cu118 | none | other
detect_current_torch_platform() {
  python - <<'PY'
try:
    import torch
except Exception:
    print("none")
else:
    cuda = torch.version.cuda
    if not cuda:
        print("cpu")
    elif str(cuda).startswith("11.8"):
        print("cu118")
    else:
        print("other")
PY
}

# Install the PyTorch stack (torch, torch-scatter, torch-geometric) for CPU or CUDA 11.8
install_pytorch_stack() {
  local platform="$1"  # cpu | cu118

  local TORCH_VERSION="2.4.1"
  local TORCH_GEOMETRIC_VERSION="2.6.1"
  local TORCH_SCATTER_VERSION="2.1.2"

  local TORCH_INDEX_CPU="https://download.pytorch.org/whl/cpu"
  local TORCH_INDEX_CU118="https://download.pytorch.org/whl/cu118"
  local PYG_SCATTER_INDEX_CPU="https://data.pyg.org/whl/torch-2.4.1+cpu.html"
  local PYG_SCATTER_INDEX_CU118="https://data.pyg.org/whl/torch-2.4.1+cu118.html"

  local torch_index pytorch_scatter_index
  case "$platform" in
    cpu)   torch_index="$TORCH_INDEX_CPU";   pytorch_scatter_index="$PYG_SCATTER_INDEX_CPU" ;;
    cu118) torch_index="$TORCH_INDEX_CU118"; pytorch_scatter_index="$PYG_SCATTER_INDEX_CU118" ;;
    *) echo "ERROR: Unknown platform '$platform'. Use 'cpu' or 'cu118'."; return 1 ;;
  esac

  echo "==> Installing PyTorch stack ($platform)…"

  # Always force reinstall all three to ensure exact versions & matching build
  echo "Installing torch==${TORCH_VERSION} from ${torch_index}…"
  pip install --no-input --no-cache-dir "torch==${TORCH_VERSION}" --index-url "${torch_index}"

  echo "Installing torch-scatter==${TORCH_SCATTER_VERSION} from ${pytorch_scatter_index}…"
  pip install --no-input --no-cache-dir "torch-scatter==${TORCH_SCATTER_VERSION}" -f "${pytorch_scatter_index}"

  echo "Installing torch-geometric==${TORCH_GEOMETRIC_VERSION}…"
  pip install --no-input --no-cache-dir "torch-geometric==${TORCH_GEOMETRIC_VERSION}"

  # Quick sanity print
  python - <<'PY'
import torch
print("torch:", torch.__version__, "CUDA:", torch.version.cuda or "cpu")
try:
    import torch_geometric, torch_scatter
    print("torch-geometric:", torch_geometric.__version__)
    print("torch-scatter:", torch_scatter.__version__)
except Exception as e:
    print("Sanity check warning:", e)
PY
}

# --- Configuration ---
ENV_NAME="packerl"
YAML_FILE="environment.yaml"

# === Main Script ===

recreate_conda_env=false
rebuild_ns3=false
debug_mode=false
install_magnneto=false
pytorch_platform_arg=""   # empty => unspecified (don’t touch torch unless env is new/recreated)

# parse options
# Leading ':' enables "silent" errors so we can handle missing args (':') and invalid opts ('?') ourselves.
# Only -p takes an argument (note the trailing ':').
while getopts ":erdmhp:" opt; do
  case "$opt" in
    e) recreate_conda_env=true ;;
    r) rebuild_ns3=true ;;
    d) debug_mode=true ;;
    m) install_magnneto=true ;;
    p) pytorch_platform_arg="$OPTARG" ;;  # expects "cpu" or "cu118"
    h) display_help; exit 0 ;;
    :) echo "ERROR: Option -$OPTARG requires an argument (cpu|cu118)." >&2; exit 1 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

# Validate -p if provided
if [ -n "$pytorch_platform_arg" ] && [ "$pytorch_platform_arg" != "cpu" ] && [ "$pytorch_platform_arg" != "cu118" ]; then
  echo "ERROR: -p requires 'cpu' or 'cu118'." >&2
  exit 1
fi

bootstrap_conda || exit 1
conda activate base || exit 1

env_created=false

# --- 1. Conda Environment Setup (create if missing, or recreate with -e) ---
if $recreate_conda_env; then
  echo "Force rebuild of conda env enabled. Removing any existing '$ENV_NAME'..."
  if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    conda env remove -n "$ENV_NAME" --yes || exit 1
  fi
  echo "Creating '$ENV_NAME' from $YAML_FILE…"
  conda env create -n "$ENV_NAME" -f "$YAML_FILE" || exit 1
  conda activate "$ENV_NAME" || exit 1
  env_created=true
else
  if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    conda activate "$ENV_NAME" || exit 1
  else
    echo "Conda environment '$ENV_NAME' not found. Creating from $YAML_FILE…"
    conda env create -n "$ENV_NAME" -f "$YAML_FILE" || exit 1
    conda activate "$ENV_NAME" || exit 1
    env_created=true
  fi
fi

# --- 2. PyTorch Stack Installation Decision ---
should_install_torch=false
desired_platform="cpu"  # default for fresh env if -p not provided

# Detect what we have right now (cpu | cu118 | none | other)
current_platform="$(detect_current_torch_platform)"

# Install torch stack if:
#  - the env was just created/recreated, OR
#  - torch is missing ("none"), OR
#  - -p is explicitly given and differs from what’s installed
if $env_created || [ "$current_platform" = "none" ]; then
  desired_platform="${pytorch_platform_arg:-cpu}"
  should_install_torch=true
  if $env_created; then
    echo "Fresh conda env created; installing PyTorch stack (${desired_platform})."
  else
    echo "PyTorch not found in existing env; installing PyTorch stack (${desired_platform})."
  fi
else
  # Existing env and torch is present: only switch if -p explicitly given and differs
  if [ -n "$pytorch_platform_arg" ]; then
    desired_platform="$pytorch_platform_arg"
    echo "Current torch platform: ${current_platform}; desired: ${desired_platform}"
    if [ "$current_platform" != "$desired_platform" ]; then
      should_install_torch=true
    else
      echo "PyTorch platform already '${current_platform}'. Skipping reinstall."
    fi
  else
    echo "PyTorch already present (${current_platform}). No -p given; leaving as-is."
  fi
fi

if $should_install_torch; then
  install_pytorch_stack "$desired_platform" || exit 1
fi

# --- 3. Install Tensorflow If Needed ---
if $install_magnneto; then
  echo "Installing TensorFlow for MAGNNETO baseline (tensorflow==2.18.1)…"
  pip install --no-input --no-cache-dir tensorflow==2.18.1
fi

# --- 4. NS-3 Setup ---

# checkout the correct branch for ns3-ai
ns3aibranch=$(git config -f .gitmodules submodule.ns3-ai.branch)
pushd ns3-ai/
  git switch $ns3aibranch
popd

# remove ns3-packerl scratch if rebuilding
if $rebuild_ns3; then
  rm -rf ns3/scratch/ns3-packerl
fi

# copy custom ns3 modules to their place in ns3
cp -r ns3-ai ns3/contrib
cp -r ns3-packerl ns3/scratch

# install ns3-ai and ns3
ns3branch=$(git config -f .gitmodules submodule.ns3.branch)
pushd ns3
  git switch $ns3branch
  pushd contrib/ns3-ai/py_interface/
    pip install -e .
  popd
  if $rebuild_ns3; then
    ./ns3 clean
  fi
  if $debug_mode; then
    ./ns3 configure -d default --enable-asserts --enable-logs
  else
    ./ns3 configure -d optimized --enable-asserts
  fi
  ./ns3 build
popd

#!/bin/bash
# Run ONCE on a Sol LOGIN node (it has internet; compute nodes do not): builds the conda env, records its Python
# interpreter for the batch jobs, and saves the Orb-v3 weights inside this folder. Safe to rerun.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/sol_config.sh"
module load "$MAMBA_MODULE"
if ! conda env list | grep -q "^$ENV_NAME "; then
    mamba env create -y -f "$here/env.yml"   # -y: no confirmation prompt to answer
fi
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null || true
conda activate "$ENV_NAME" 2>/dev/null || source activate "$ENV_NAME" 2>/dev/null || true
if ! python -c "import orb_models" 2>/dev/null; then
    echo "ERROR: environment $ENV_NAME is not active (python: $(command -v python || echo none))"; exit 2
fi
# The phonon test needs phonopy; an environment built before it was added gets it here.
python -c "import phonopy" 2>/dev/null || pip install -q "phonopy==4.8.3"
# The jobs call this interpreter directly, so whatever environment the submitting shell has active cannot leak in.
python -c "import sys; print(sys.executable)" > "$here/env_python.txt"
mkdir -p "$here/weights"
python - "$here/weights" <<'PY'
import inspect, shutil, sys
from pathlib import Path
from cached_path import cached_path
from orb_models.forcefield import pretrained
# The URL is read from orb-models itself. A local file path makes cached_path skip the network check it otherwise
# makes on every load, which on Sol's offline compute nodes cost 90 minutes of timeouts per run.
# Both models: the 20-neighbour one of the first sweep and the uncapped one of the rerun.
for name in ("orb_v3_conservative_20_omat", "orb_v3_conservative_inf_omat"):
    loader = getattr(pretrained, name)
    url = inspect.signature(loader).parameters["weights_path"].default
    dest = Path(sys.argv[1]) / url.rsplit("/", 1)[1]
    if not dest.exists():
        shutil.copy(cached_path(url), dest)
    loader(weights_path=str(dest), device="cpu", precision="float32-high", compile=False)
    print("weights in", dest.name)
import torch, ase, orb_models, phonopy
print("torch", torch.__version__, "| ase", ase.__version__, "| orb-models",
      __import__("importlib.metadata").metadata.version("orb-models"), "| phonopy", phonopy.__version__)
PY
echo "interpreter: $(cat "$here/env_python.txt")"
echo "SETUP OK"

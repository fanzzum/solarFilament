from pathlib import Path
import ast
import copy
import json
import textwrap
import uuid

path = Path('solarcrow7.ipynb')
nb = json.loads(path.read_text())
original = copy.deepcopy(nb['cells'])
assert len(original) == 68

def source(index):
    return ''.join(original[index]['source'])

def clean(text):
    return textwrap.dedent(text).strip() + '\n'

def code(text, name):
    return {'cell_type': 'code', 'source': clean(text), 'metadata': {'name': name, 'tags': ['q0-q4', name]}, 'execution_count': None, 'outputs': [], 'id': uuid.uuid4().hex[:12]}

def md(text):
    return {'cell_type': 'markdown', 'source': clean(text), 'metadata': {'tags': ['q0-q4']}, 'id': uuid.uuid4().hex[:12]}

def old(index, text=None):
    cell = copy.deepcopy(original[index])
    cell['metadata']['q0_q4_original_cell_index'] = index
    if text is not None:
        cell['source'] = clean(text)
        # Retain any historical outputs; a changed source has not been executed.
        cell['execution_count'] = None
        cell['metadata']['q0_q4_source_changed_outputs_historical'] = True
    return cell

def gated(text, flag):
    return f'if ACTIONS[{flag!r}]:\n' + textwrap.indent(clean(text), '    ') + f'else:\n    print({flag!r} + " skipped (opt-in).")\n'

cells = []
cells.append(md('''
# Solar Filament Segmentation — Q0–Q4 verification notebook

## 1. Overview and execution safety

This is the existing **ResNet34 coarse U-Net → region-conditioned fine U-Net** solution, not a new architecture. Q0–Q4 repairs separate checkpoint reuse, intentional training, inference, evaluation and diagnostics.

**Historical references, not rerun here:** local PQ approximately **0.322** (old detailed evaluator; simple scorer 0.3227), public leaderboard **0.29** from a separate submitted artifact. Their checkpoint/version association is unresolved. Pairwise A3 PQ 0.3397 is an annotation-ambiguity reference, not a ceiling; corrected matching may change it. Current source previously trained coarse, then replaced it with an external coarse checkpoint, loaded fine, then recreated/trained fine. That accidental ownership chain is removed below.

**Nothing has been executed or verified during this refactor.** Existing checkpoints remain valid inference inputs; the corrected loss does not retroactively change their training. Original cell metadata/outputs, if present, are retained and may describe historical source, not a successful refactored run. This supplied notebook has no retained code outputs; historical numbers come from the accompanying notes.

### Safe first pass

Run configuration/imports, dataset indexing, fixed-split checks and utility definitions. Leave all `ACTIONS` false. Enable only `synthetic_tests` for the prepared CPU tests, then inspect checkpoint/config paths and explicitly enable `load_models` (no training). CSV auditing can inspect an existing CSV without generating a new one. Validation inference is a **separate opt-in action** after those checks.

Training requires **both** a stage mode (`train` or `finetune`) **and** its explicit action. Installation, previews, full A3, model inference, artifact writes and submission generation are off by default. Never use “Run all” with training/export actions enabled unless you intend those operations. Sections are ordered for a fresh kernel; repeated training is refused until you deliberately reset the selected in-memory stage.
'''))
cells.append(md('''
## 2. Imports and environment

Reuse the existing packages. Dependency installation is optional and can change the environment; it is disabled by default. The original installation command is retained. No package upgrades, training or inference are needed for static review. Device defaults to the original effective **CUDA/CPU** choice; MPS is available only by explicit request.
'''))
cells.append(code('''
INSTALL_DEPENDENCIES = False
if INSTALL_DEPENDENCIES:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch'])
''', 'optional-install'))
imports = source(3).replace('# All imports in one place (cv2 was used further down but never imported)', '# Shared imports; utility definitions must not depend on a training cell.')
imports += '''\nimport copy\nimport hashlib\nimport importlib.metadata\nimport platform\nimport re\nimport uuid\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nimport torch.nn.functional as F\n'''
cells.append(old(3, imports))
env = source(4)
env = env.replace('os.makedirs(working_dir, exist_ok=True)', '# Reading setup never creates or overwrites an artifact directory.')
start = env.index('# 2. Hardware:')
end = env.index("print(f'Active Device", start)
env = env[:start] + '''# Resolve once; explicit DEVICE_NAME='mps' is optional, not a silent policy change.
DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(DEVICE_NAME)

''' + env[end:]
cells.append(old(4, env))
cells.append(md('''
### 2.1 Compact configuration — edit here, not in training/inference helpers

`load` reuses a checkpoint; `train` starts from the original ImageNet initialization; `finetune` loads the selected checkpoint **before** constructing the original optimizer. Fine-tuning starts a new optimizer (it is not an exact optimizer-state resume). Change checkpoint paths here to select another pair, call `reset_selected_models()` explicitly, then rerun loading; do not rerun training to compare pairs.

The Kaggle paths below are the existing inputs. On a local machine set them explicitly, e.g. `models/coarse_model.pth` and `models/fine_model2.pth`; no automatic checkpoint substitution occurs. Set `SPLIT_PATH` to the saved split mounted on Kaggle if it is not in the working directory. Missing split creation is opt-in, deterministic and exclusive; it does **not** reconstruct unknown historical membership.

Inference settings are independent of training settings. The default reference bucket sizes/settings are the existing `outputs/model_config.json` values, **not evidence that an external checkpoint used them**. If a frozen config file exists at `INFERENCE_CONFIG_PATH`, it is loaded read-only. Training-derived buckets remain available for new fine training, without changing inference settings behind your back.
'''))
cells.append(code('''
ACTIONS = {
    'load_models': False,
    'train_coarse': False,
    'build_fine_samples': False,
    'train_fine': False,
    'synthetic_tests': False,
    'eda': False,
    'previews': False,
    'a3_analysis': False,
    'validation': False,
    'coarse_diagnostic': False,
    'coarse_only_evaluation': False,
    'coverage_diagnostic': False,
    'q4_diagnostics': False,
    'submission': False,
    'csv_validation': False,
    'write_artifacts': False,
}
MODEL_SETTINGS = {
    'coarse': {
        'mode': 'load',  # load | train | finet
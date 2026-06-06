# Two-Stage NLG Training (Context → Meaning)

This repository contains a two-stage training + evaluation pipeline for text generation.

- **Stage A**: Given `processed_sentence` → predict `Context_New`
- **Stage B**: Given `processed_sentence` + predicted context → predict `Meaning_New`

The script also:
- records runtime environment to `environment.json`
- saves per-stage `train_meta.json`
- outputs checkpoints under `stageA_context/` and `stageB_meaning/`
- evaluates with multiple metrics (BLEU/ROUGE/BERTScore/WER/etc.)
- computes additional metrics split by polysemy/ambiguity of keys

## Code

- `src/train_two_stage.py`  
  Main training + evaluation entry point.

## Data (repo root)

The script expects your dataset CSV at the repo root:

- `./Updated_Training_Dataset_Cultural_Responses_half.csv`

The CSV must include at least these columns:
- `sentence`
- `Meaning_New`

The script derives:
- `processed_sentence`
- `Context_New` from `<UGE>...</UGE>` spans inside `sentence`

## Install

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

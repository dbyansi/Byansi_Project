import os
import sys
import time
import json
import random
import logging
import platform
import socket
import re
from collections import defaultdict
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np
import torch

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Metric libraries
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sacrebleu import corpus_chrf
import bert_score
from jiwer import wer
from nltk.metrics import edit_distance
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Logging
# =========================
log_file_path = "NER_Training_Metrics_CCR3_1_1_1_1_20.log"
handlers = [logging.StreamHandler(sys.stdout)]
try:
    handlers.append(logging.FileHandler(log_file_path))
except OSError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
log = logging.getLogger(__name__)

# =========================
# Determinism / seeds
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

# =========================
# Utility: record environment
# =========================
def record_environment(output_dir: str):
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": os.environ.get("CUDA_VERSION", "unknown"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(os.path.join(output_dir, "environment.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    log.info(f"Environment recorded to {output_dir}/environment.json")

# =========================
# Gloss-only enforcement
# =========================
def gloss_only_text(s: str) -> str:
    """
    Extract "gloss-only" target text from your Meaning_New values.
    """
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""

    s = " ".join(s.split())

    if re.search(r"\bmeaning\b", s, flags=re.IGNORECASE):
        quoted = re.findall(r"'([^']+)'", s)
        if quoted:
            s = quoted[-1].strip()
        else:
            s = re.split(r"\bmeaning\b", s, flags=re.IGNORECASE)[0].strip()

    s = re.sub(r"\s+([,.;!?])", r"\1", s)
    s = s.strip(" ;,.")

    s = " ".join(s.split())
    return s

def extract_gloss_from_generated(s: str) -> str:
    return gloss_only_text(s)

# =========================
# Deterministic Stage A context
# =========================
def extract_uge_spans_exact_concatenation(original_sentence: str) -> str:
    if original_sentence is None:
        return ""
    s = str(original_sentence)
    spans = re.findall(r"<UGE>(.*?)</UGE>", s, flags=re.DOTALL)
    spans = ["" if sp is None else str(sp) for sp in spans]
    spans = [sp for sp in spans if sp.strip() != ""]
    return " ".join(spans).strip()

def build_pred_contexts_deterministic(test_df: pd.DataFrame) -> List[str]:
    if "original_sentence" in test_df.columns:
        source_col = "original_sentence"
    else:
        source_col = "sentence" if "sentence" in test_df.columns else "processed_sentence"
    return [extract_uge_spans_exact_concatenation(x) for x in test_df[source_col].tolist()]

# =========================
# Data loading & processing
# =========================
def load_data(file_path: str) -> pd.DataFrame:
    log.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, encoding="latin1", low_memory=False)
    log.info(f"Loaded {len(df)} rows")
    return df

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["original_sentence"] = df["sentence"].fillna("").astype(str)
    df["Context_New_list"] = df["original_sentence"].apply(lambda x: re.findall(r"<UGE>(.*?)</UGE>", str(x)))

    def concat_all_context_spans(spans):
        if not isinstance(spans, list) or len(spans) == 0:
            return ""
        spans = ["" if s is None else str(s).strip() for s in spans]
        spans = [s for s in spans if s]
        return " ".join(spans)

    df["Context_New"] = df["Context_New_list"].apply(concat_all_context_spans)

    df["processed_sentence"] = df["original_sentence"].apply(
        lambda x: re.sub(r"<UGE>(.*?)</UGE>", r"\1", str(x)).strip()
    )

    df["Meaning_New"] = df["Meaning_New"].fillna("").astype(str).apply(gloss_only_text)

    log.info(
        "After processing, sample rows:\n"
        f"{df[['original_sentence','processed_sentence','Context_New','Meaning_New']].head()}"
    )
    return df

def collate_fn_strip_context(data, hf_collator):
    stripped = []
    for x in data:
        x = dict(x)
        x.pop("context_token", None)
        stripped.append(x)
    return hf_collator(stripped)

# =========================
# Dataset
# =========================
class TextDataset(Dataset):
    def __init__(
        self,
        inputs_encodings: Dict[str, List[List[int]]],
        labels_encodings: List[List[int]],
        context_tokens: List[str],
        tokenizer: AutoTokenizer,
        is_causal: bool = False,
        prompt_prefix: str = "",
    ):
        self.inputs_encodings = inputs_encodings
        self.labels_encodings = labels_encodings
        self.context_tokens = context_tokens
        self.tokenizer = tokenizer
        self.is_causal = is_causal
        self.prompt_prefix = prompt_prefix

    def __len__(self):
        return len(self.labels_encodings)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.inputs_encodings.items()}
        item_labels = torch.tensor(self.labels_encodings[idx])
        item["labels"] = item_labels
        item["context_token"] = self.context_tokens[idx]
        return item

# =========================
# Tokenization helpers
# =========================
def tokenize_prompts_and_labels(
    *,
    tokenizer: AutoTokenizer,
    texts: List[str],
    ctxs: List[str],
    targets: List[str],
    max_length: int,
    is_causal: bool,
    prompt_template: str,
) -> Tuple[Dict[str, List[List[int]]], List[List[int]]]:
    prompts = []
    for s, c in zip(texts, ctxs):
        prompts.append(prompt_template.format(sent=s, ctx=c if c is not None else "").strip())

    inputs = tokenizer(
        prompts,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_attention_mask=True,
    )

    targets = [(t).strip() if t is not None else "" for t in targets]
    labels_tok = tokenizer(
        targets,
        truncation=True,
        padding="longest",
        max_length=max_length,
    )
    labels_input_ids = labels_tok["input_ids"]

    if is_causal:
        concatenated_input_ids = []
        concatenated_attention_masks = []
        concatenated_labels = []

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for inp_ids, lbl_ids in zip(inputs["input_ids"], labels_input_ids):
            concat = (inp_ids + lbl_ids)[:max_length]
            label_ids = ([-100] * len(inp_ids) + lbl_ids)[:max_length]

            attn = [1] * len(concat)
            pad_len = max_length - len(concat)
            if pad_len > 0:
                concat = concat + [pad_id] * pad_len
                attn = attn + [0] * pad_len
                label_ids = label_ids + [-100] * pad_len

            concatenated_input_ids.append(concat)
            concatenated_attention_masks.append(attn)
            concatenated_labels.append(label_ids)

        return {"input_ids": concatenated_input_ids, "attention_mask": concatenated_attention_masks}, concatenated_labels

    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}, labels_input_ids

# =========================
# Model loading helper
# =========================
def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token

    try:
        config = AutoConfig.from_pretrained(model_name)
        is_encoder_decoder = getattr(config, "is_encoder_decoder", None)
        if is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            return model, tokenizer, False
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            return model, tokenizer, False
        except Exception:
            pass
    except Exception:
        log.warning(f"Could not determine encoder-decoder property for {model_name}; trying causal LM fallback.")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer, True

# =========================
# Metrics (with per-example support)
# =========================
sbert = SentenceTransformer("all-MiniLM-L6-v2")
smoothing_function = SmoothingFunction().method1
rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def compute_retrieval_eval(true: str, pred: str):
    true_words = set(true.split())
    pred_words = set(pred.split())
    lexical_overlap = len(true_words.intersection(pred_words)) / max(len(true_words), 1)

    emb = sbert.encode([true, pred])
    semantic_score = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return (lexical_overlap + semantic_score) / 2

def calculate_rouge(true: str, pred: str):
    scores = rouge_scorer_obj.score(true, pred)
    return {"rouge1": scores["rouge1"].fmeasure, "rouge2": scores["rouge2"].fmeasure, "rougeL": scores["rougeL"].fmeasure}

def single_pair_metrics(ref: str, pred: str) -> Dict[str, float]:
    ref = "" if ref is None else str(ref)
    pred = "" if pred is None else str(pred)

    ref_toks = ref.split()
    pred_toks = pred.split()

    bleu = sentence_bleu([ref_toks], pred_toks, smoothing_function=smoothing_function)
    rouge = calculate_rouge(ref, pred)
    ter = edit_distance(ref_toks, pred_toks) / max(len(ref_toks), 1)
    ed = edit_distance(ref_toks, pred_toks)

    w = wer(ref, pred)

    try:
        chrf = corpus_chrf([ref], [pred]).score
    except Exception:
        chrf = 0.0

    retev = compute_retrieval_eval(ref, pred)

    try:
        _, _, F = bert_score.score([pred], [ref], lang="en", verbose=False)
        bert_f = F[0].item()
    except Exception:
        bert_f = 0.0

    return {
        "BLEU": float(bleu),
        "ROUGE1": float(rouge["rouge1"]),
        "ROUGE2": float(rouge["rouge2"]),
        "ROUGEL": float(rouge["rougeL"]),
        "TER": float(ter),
        "BERTScore": float(bert_f),
        "EditDistance": float(ed),
        "WER": float(w),
        "chrF": float(chrf),
        "RetriEVAL": float(retev),
    }

def compute_metrics_batch(references: List[str], predictions: List[str]) -> Dict[str, float]:
    out = {k: [] for k in [
        "BLEU","ROUGE1","ROUGE2","ROUGEL","TER","BERTScore","EditDistance","WER","chrF","RetriEVAL"
    ]}

    for ref, pred in zip(references, predictions):
        m = single_pair_metrics(ref, pred)
        for k, v in m.items():
            out[k].append(v)

    return {k: float(np.mean(v)) if len(v) > 0 else None for k, v in out.items()}

# =========================
# Polysemy (multi-reference aware; mean/soft)
# =========================
def normalize_key_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = " ".join(s.split())
    return s

def build_polysemy_maps(test_df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    references_map = defaultdict(set)
    for _, row in test_df.iterrows():
        key = normalize_key_text(row["processed_sentence"])
        meaning = row["Meaning_New"]
        references_map[key].add(meaning)
    references_map = {k: list(v) for k, v in references_map.items()}
    distinct_count_map = {k: len(v) for k, v in references_map.items()}
    return references_map, distinct_count_map

def polysemy_aggregate(values: List[float], mode: str = "mean", alpha: float = 5.0) -> float:
    if len(values) == 0:
        return float("nan")
    if mode == "mean":
        return float(np.mean(values))
    if mode == "softmax_mean":
        arr = np.array(values, dtype=np.float64)
        mx = np.max(arr)
        ex = np.exp((arr - mx) * alpha)
        w = ex / (np.sum(ex) + 1e-12)
        return float(np.sum(w * arr))
    return float(np.mean(values))

def compute_metrics_batch_with_polysemy_per_example(
    test_df: pd.DataFrame,
    predictions: List[str],
    task_col: str,
    references_map: Dict[str, List[str]],
    polysemy_aggregation: str = "mean",
    polysemy_soft_alpha: float = 5.0,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    metrics = ["BLEU","ROUGE1","ROUGE2","ROUGEL","TER","BERTScore","EditDistance","WER","chrF","RetriEVAL"]
    per_example_rows = []
    metric_lists = {k: [] for k in metrics}

    for i, pred in enumerate(predictions):
        key = normalize_key_text(test_df.iloc[i]["processed_sentence"])
        gold_candidates = references_map.get(key, [])
        if not gold_candidates:
            gold_candidates = ["" if task_col not in test_df.columns else str(test_df.iloc[i][task_col])]

        cand_metrics = []
        for ref in gold_candidates:
            m = single_pair_metrics(ref, pred)
            cand_metrics.append(m)

        agg_m = {}
        for k in metrics:
            vals = [cm[k] for cm in cand_metrics]
            agg_m[k] = polysemy_aggregate(vals, mode=polysemy_aggregation, alpha=polysemy_soft_alpha)

        for k in metrics:
            metric_lists[k].append(agg_m[k])

        per_example_rows.append({
            "index": int(test_df.index[i]) if "index" in test_df.columns else int(i),
            "key": key,
            "n_gold_candidates": len(gold_candidates),
            "gold_candidates": " ||| ".join([str(x) for x in gold_candidates]),
            "predicted": pred,
            **{f"m_poly_{k}": agg_m[k] for k in metrics},
        })

    aggregated = {k: float(np.mean(metric_lists[k])) if len(metric_lists[k]) > 0 else None for k in metrics}
    per_example_df = pd.DataFrame(per_example_rows)
    return aggregated, per_example_df

# =========================
# Train one stage
# =========================
def train_single_stage(
    *,
    model_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    stage_output_dir: str,
    prompt_template: str,
    target_col: str,
    ctx_col_for_inputs: Optional[str],
    max_length: int = 256,
    num_epochs: int = 2,
    per_device_train_batch_size: int = 8,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    warmup_steps: int = 100,
    seed: int = 42,
    early_stopping_patience: int = 3,
) -> Tuple[str, str]:
    set_seed(seed)
    model, tokenizer, is_causal = load_model_and_tokenizer(model_name, device)

    os.makedirs(stage_output_dir, exist_ok=True)
    record_environment(stage_output_dir)

    train_meta = {
        "model_name": model_name,
        "is_causal": is_causal,
        "num_epochs": num_epochs,
        "batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "seed": seed,
        "prompt_template": prompt_template,
        "target_col": target_col,
        "ctx_col_for_inputs": ctx_col_for_inputs,
    }
    with open(os.path.join(stage_output_dir, "train_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(train_meta, fh, indent=2)

    train_texts = train_df["processed_sentence"].tolist()
    valid_texts = valid_df["processed_sentence"].tolist()

    train_targets = train_df[target_col].tolist()
    valid_targets = valid_df[target_col].tolist()

    if target_col == "Meaning_New":
        train_targets = [gloss_only_text(x) for x in train_targets]
        valid_targets = [gloss_only_text(x) for x in valid_targets]

    if ctx_col_for_inputs is None:
        train_ctxs = [""] * len(train_texts)
        valid_ctxs = [""] * len(valid_texts)
    else:
        train_ctxs = train_df[ctx_col_for_inputs].tolist()
        valid_ctxs = valid_df[ctx_col_for_inputs].tolist()

    inputs_train, labels_train = tokenize_prompts_and_labels(
        tokenizer=tokenizer,
        texts=train_texts,
        ctxs=train_ctxs,
        targets=train_targets,
        max_length=max_length,
        is_causal=is_causal,
        prompt_template=prompt_template,
    )
    inputs_valid, labels_valid = tokenize_prompts_and_labels(
        tokenizer=tokenizer,
        texts=valid_texts,
        ctxs=valid_ctxs,
        targets=valid_targets,
        max_length=max_length,
        is_causal=is_causal,
        prompt_template=prompt_template,
    )

    train_dataset = TextDataset(inputs_train, labels_train, train_ctxs, tokenizer, is_causal=is_causal)
    valid_dataset = TextDataset(inputs_valid, labels_valid, valid_ctxs, tokenizer, is_causal=is_causal)

    if not is_causal:
        hf_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")
    else:
        hf_collator = DataCollatorWithPadding(tokenizer, padding="longest")

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_strip_context(batch, hf_collator),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_strip_context(batch, hf_collator),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_valid_loss = float("inf")
    epochs_no_improve = 0

    val_losses_csv = os.path.join(stage_output_dir, "validation_losses.csv")
    if not os.path.exists(val_losses_csv):
        pd.DataFrame(columns=["epoch", "valid_loss"]).to_csv(val_losses_csv, index=False, encoding="utf-8")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        start = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items() if k != "context_token"}

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                log.info(
                    f"{os.path.basename(stage_output_dir)} | Epoch {epoch+1}/{num_epochs} "
                    f"| Step {step+1}/{len(train_loader)} | Loss {loss.item():.4f}"
                )

        avg_train_loss = total_loss / max(1, len(train_loader))
        log.info(
            f"{os.path.basename(stage_output_dir)} | Epoch {epoch+1} finished. "
            f"Avg train loss: {avg_train_loss:.4f}. Took {time.time()-start:.1f}s"
        )

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                batch_device = {k: v.to(device) for k, v in batch.items() if k != "context_token"}
                outputs = model(**batch_device)
                valid_loss += outputs.loss.item()

        avg_valid_loss = valid_loss / max(1, len(valid_loader))
        log.info(f"{os.path.basename(stage_output_dir)} | Validation loss after epoch {epoch+1}: {avg_valid_loss:.4f}")

        df_val_row = pd.DataFrame([{"epoch": epoch + 1, "valid_loss": avg_valid_loss}])
        df_val_row.to_csv(val_losses_csv, index=False, mode="a", header=False, encoding="utf-8")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            epochs_no_improve = 0
            best_ckpt = os.path.join(stage_output_dir, "best_checkpoint")
            model.save_pretrained(best_ckpt)
            tokenizer.save_pretrained(best_ckpt)
            log.info(f"Saved best checkpoint to {best_ckpt}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                log.info(f"{os.path.basename(stage_output_dir)} | Early stopping triggered.")
                break

    best_ckpt_dir = os.path.join(stage_output_dir, "best_checkpoint")
    return best_ckpt_dir, best_ckpt_dir

# =========================
# Inference
# =========================
def generate_stage_outputs(
    *,
    checkpoint_dir: str,
    model_kind_name: str,
    stage_prompt_template: str,
    texts: List[str],
    ctxs: List[str],
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    is_causal: bool,
    max_length: int,
    batch_size: int = 8,
    num_beams: int = 4,
) -> List[str]:
    model.eval()
    results = []

    prompts = [
        stage_prompt_template.format(sent=s, ctx=(c if c is not None else "")).strip()
        for s, c in zip(texts, ctxs)
    ]

    enc = tokenizer(
        prompts,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "early_stopping": True}

    with torch.no_grad():
        n = enc["input_ids"].shape[0]
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            batch_enc = {k: v[start:end] for k, v in enc.items()}
            out_ids = model.generate(**batch_enc, **gen_kwargs)

            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            if model_kind_name == "stageB":
                decoded = [extract_gloss_from_generated(x) for x in decoded]

            results.extend(decoded)

    return results

def load_from_checkpoint(checkpoint_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir).to(device)
        return model, tokenizer, False
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir).to(device)
        return model, tokenizer, True

# =========================
# Evaluation (multi-condition + extra CSVs)
# =========================
def evaluate_two_stage(
    *,
    model_name: str,
    test_df: pd.DataFrame,
    stageA_ckpt_dir: str,   # kept for signature compatibility
    stageB_ckpt_dir: str,
    output_dir: str,
    max_length: int,
    batch_size: int = 8,
    num_beams: int = 4,
    polysemy_aggregation: str = "mean",
    polysemy_soft_alpha: float = 5.0,
    # NEW: pass global maps (computed on full dataset)
    references_map_meaning: Optional[Dict[str, List[str]]] = None,
    distinct_count_map_meaning: Optional[Dict[str, int]] = None,
):
    modelB, tokenizerB, is_causalB = load_from_checkpoint(stageB_ckpt_dir)

    test_texts = test_df["processed_sentence"].tolist()
    gold_contexts = test_df["Context_New"].tolist()
    gold_meanings = test_df["Meaning_New"].tolist()

    # Deterministic Stage A contexts
    pred_contexts_stageA = build_pred_contexts_deterministic(test_df)

    prompt_B = (
        "{sent} Context: {ctx}\n"
        "Output ONLY the gloss/definition (no extra explanations)."
    )

    # Condition 1: Gold context -> meaning
    pred_meanings_goldctx = generate_stage_outputs(
        checkpoint_dir=stageB_ckpt_dir,
        model_kind_name="stageB",
        stage_prompt_template=prompt_B,
        texts=test_texts,
        ctxs=gold_contexts,
        tokenizer=tokenizerB,
        model=modelB,
        is_causal=is_causalB,
        max_length=max_length,
        batch_size=batch_size,
        num_beams=num_beams,
    )

    # Condition 2: Pred context (from deterministic Stage A) -> meaning
    pred_meanings_predctx = generate_stage_outputs(
        checkpoint_dir=stageB_ckpt_dir,
        model_kind_name="stageB",
        stage_prompt_template=prompt_B,
        texts=test_texts,
        ctxs=pred_contexts_stageA,
        tokenizer=tokenizerB,
        model=modelB,
        is_causal=is_causalB,
        max_length=max_length,
        batch_size=batch_size,
        num_beams=num_beams,
    )

    # Condition 3: No context ablation -> meaning
    blank_contexts = [""] * len(test_texts)
    pred_meanings_nocontext = generate_stage_outputs(
        checkpoint_dir=stageB_ckpt_dir,
        model_kind_name="stageB",
        stage_prompt_template=prompt_B,
        texts=test_texts,
        ctxs=blank_contexts,
        tokenizer=tokenizerB,
        model=modelB,
        is_causal=is_causalB,
        max_length=max_length,
        batch_size=batch_size,
        num_beams=num_beams,
    )

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Global polysemy maps used for ambiguity splitting
    # -------------------------
    if references_map_meaning is None or distinct_count_map_meaning is None:
        # fallback to old behavior (should not happen if main passes them)
        references_map_meaning, distinct_count_map_meaning = build_polysemy_maps(test_df)

    test_df_local = test_df.copy().reset_index(drop=True)
    test_df_local["key"] = [normalize_key_text(s) for s in test_df_local["processed_sentence"].tolist()]
    test_df_local["n_distinct_meanings_for_key"] = (
        test_df_local["key"].map(distinct_count_map_meaning).fillna(1).astype(int)
    )
    test_df_local["is_ambiguous_key"] = test_df_local["n_distinct_meanings_for_key"] >= 2

    ambiguous_mask = test_df_local["is_ambiguous_key"].values

    # ------------------------------------------------------------
    # Helper: compute polysemy-aware metrics for a prediction list
    # and optionally a dataframe subset (row-aligned by position).
    # ------------------------------------------------------------
    metrics_list_order = [
        "BLEU", "ROUGE1", "ROUGE2", "ROUGEL", "TER",
        "BERTScore", "EditDistance", "WER", "chrF", "RetriEVAL"
    ]

    def run_meaning_condition(pred_meanings_full: List[str], cond_name: str, df_subset: pd.DataFrame):
        if len(df_subset) == 0:
            agg_none = {k: None for k in metrics_list_order}
            return agg_none, pd.DataFrame()

        positions = df_subset.index.tolist()
        pred_sel = [pred_meanings_full[pos] for pos in positions]

        agg, per_ex_df = compute_metrics_batch_with_polysemy_per_example(
            test_df=df_subset.reset_index(drop=False),  # keep position as column "index"
            predictions=pred_sel,
            task_col="Meaning_New",
            references_map=references_map_meaning,
            polysemy_aggregation=polysemy_aggregation,
            polysemy_soft_alpha=polysemy_soft_alpha,
        )

        per_ex_path = os.path.join(output_dir, f"test_meaning_per_example_{cond_name}.csv")
        per_ex_df.to_csv(per_ex_path, index=False, encoding="utf-8")
        return agg, per_ex_df

    # Full polysemy-aware evaluation for each condition
    meaning_metrics_goldctx, _ = run_meaning_condition(
        pred_meanings_full=pred_meanings_goldctx,
        cond_name="gold_context",
        df_subset=test_df_local,
    )
    meaning_metrics_predctx, meaning_per_ex_predctx_df = run_meaning_condition(
        pred_meanings_full=pred_meanings_predctx,
        cond_name="pred_context",
        df_subset=test_df_local,
    )
    meaning_metrics_nocontext, _ = run_meaning_condition(
        pred_meanings_full=pred_meanings_nocontext,
        cond_name="no_context",
        df_subset=test_df_local,
    )

    # Ambiguous/unambiguous splits (predctx)
    test_df_amb = test_df_local[ambiguous_mask]
    test_df_unamb = test_df_local[~ambiguous_mask]

    meaning_metrics_predctx_amb, _ = run_meaning_condition(
        pred_meanings_full=pred_meanings_predctx,
        cond_name="pred_context_ambiguous",
        df_subset=test_df_amb,
    ) if len(test_df_amb) > 0 else ({k: None for k in meaning_metrics_predctx}, pd.DataFrame())

    meaning_metrics_predctx_unamb, _ = run_meaning_condition(
        pred_meanings_full=pred_meanings_predctx,
        cond_name="pred_context_unambiguous",
        df_subset=test_df_unamb,
    ) if len(test_df_unamb) > 0 else ({k: None for k in meaning_metrics_predctx}, pd.DataFrame())

    # Context metrics: gold Context_New vs deterministic predicted contexts
    context_metrics = compute_metrics_batch(gold_contexts, pred_contexts_stageA)

    # Save combined summaries (CSV) for each condition
    meaning_metrics_rows = []
    for cond, m in [
        ("gold_context", meaning_metrics_goldctx),
        ("pred_context", meaning_metrics_predctx),
        ("no_context", meaning_metrics_nocontext),
    ]:
        for k, v in m.items():
            meaning_metrics_rows.append({
                "model_name": model_name,
                "meaning_condition": cond,
                "metric": k,
                "value": v,
            })
    meaning_metrics_df = pd.DataFrame(meaning_metrics_rows)
    meaning_metrics_df.to_csv(
        os.path.join(output_dir, "test_meaning_metrics_by_condition.csv"),
        index=False, encoding="utf-8"
    )

    # Save additional per-example joined CSV (extra info)
    results_df = pd.DataFrame({
        "model_name": [model_name] * len(test_df_local),
        "index": test_df_local.index,
        "processed_sentence": test_df_local["processed_sentence"].tolist(),
        "gold_context": gold_contexts,
        "predicted_context_stageA": pred_contexts_stageA,
        "gold_meaning": gold_meanings,
        "predicted_meaning_gold_context": pred_meanings_goldctx,
        "predicted_meaning_pred_context": pred_meanings_predctx,
        "predicted_meaning_no_context": pred_meanings_nocontext,
        "key": test_df_local["key"].tolist(),
        "n_distinct_meanings_for_key": test_df_local["n_distinct_meanings_for_key"].tolist(),
        "is_ambiguous_key": test_df_local["is_ambiguous_key"].tolist(),
    })

    # Join meaning per-example metrics for pred_context
    if "index" in meaning_per_ex_predctx_df.columns:
        meaning_per_ex_predctx_df2 = meaning_per_ex_predctx_df.copy()
        meaning_per_ex_predctx_df2 = meaning_per_ex_predctx_df2.rename(columns={"index": "result_index"})
        results_df = results_df.merge(
            meaning_per_ex_predctx_df2,
            left_on="index",
            right_on="result_index",
            how="left",
            suffixes=("", "_poly"),
        )
        if "result_index" in results_df.columns:
            results_df.drop(columns=["result_index"], inplace=True)

    results_path = os.path.join(output_dir, "test_predictions_two_stage.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8")

    # Save context metrics
    context_metrics_df = pd.DataFrame(
        [{"model_name": model_name, "task": "context", "metric": k, "value": v} for k, v in context_metrics.items()]
    )
    context_metrics_df.to_csv(os.path.join(output_dir, "test_context_metrics.csv"), index=False, encoding="utf-8")

    # Backward compatibility combined summary (pred_context as main)
    meaning_metrics_main = meaning_metrics_predctx
    combined_summary = []
    for k, v in meaning_metrics_main.items():
        combined_summary.append({"model_name": model_name, "task": "meaning_pred_context", "metric": k, "value": v})
    for k, v in context_metrics.items():
        combined_summary.append({"model_name": model_name, "task": "context", "metric": k, "value": v})

    pd.DataFrame(combined_summary).to_csv(
        os.path.join(output_dir, "test_metrics_summary_two_stage1.csv"),
        index=False, encoding="utf-8"
    )

    # Save ambiguous/unambiguous summary for pred_context
    meaning_split_rows = []
    for split_name, m in [
        ("ambiguous", meaning_metrics_predctx_amb),
        ("unambiguous", meaning_metrics_predctx_unamb),
    ]:
        for k, v in m.items():
            meaning_split_rows.append({
                "model_name": model_name,
                "task": f"meaning_pred_context_{split_name}",
                "metric": k,
                "value": v
            })
    pd.DataFrame(meaning_split_rows).to_csv(
        os.path.join(output_dir, "test_meaning_metrics_ambig_split_pred_context.csv"),
        index=False, encoding="utf-8"
    )

    return meaning_metrics_main, context_metrics

# =========================
# MAIN
# =========================
def main():
    try:
        output_dir = "./results/culturally_contextualized_responses_two_stage_2026_06_30"
        os.makedirs(output_dir, exist_ok=True)

        data = load_data("Updated_Training_Dataset_Cultural_Responses_half.csv")
        data = process_data(data)

        data = data[data["processed_sentence"].astype(bool) & data["Meaning_New"].astype(bool)]
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
        valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        log.info(f"Splits : train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

        # =========================
        # NEW: compute polysemy/ambiguity from full dataset BEFORE evaluating ambiguity splits
        # =========================
        full_df_for_ambiguity = data.copy().reset_index(drop=True)
        references_map_meaning, distinct_count_map_meaning = build_polysemy_maps(full_df_for_ambiguity)
        log.info(f"Computed global polysemy maps from full dataset: {len(full_df_for_ambiguity)} rows")
        log.info(f"Num distinct keys in global map: {len(distinct_count_map_meaning)}")

        model_names = ["Helsinki-NLP/opus-mt-mul-en"]

        summary_metrics_csv = os.path.join(output_dir, "all_models_test_metrics_summary_two_stage1.csv")
        summary_rows = []

        for model_name in model_names:
            try:
                log.info(f"Running two-stage training/eval for {model_name}")

                base_model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
                stageA_dir = os.path.join(base_model_dir, "stageA_context")
                stageB_dir = os.path.join(base_model_dir, "stageB_meaning")

                # Stage A training (kept, though not used for inference)
                stageA_prompt_template = "{sent}"
                stageA_ckpt_dir, _ = train_single_stage(
                    model_name=model_name,
                    train_df=train_df,
                    valid_df=valid_df,
                    stage_output_dir=stageA_dir,
                    prompt_template=stageA_prompt_template,
                    target_col="Context_New",
                    ctx_col_for_inputs=None,
                    max_length=256,
                    num_epochs=5,
                    per_device_train_batch_size=8,
                    learning_rate=5e-5,
                    warmup_steps=50,
                    early_stopping_patience=2,
                )

                # Stage B training
                stageB_prompt_template = (
                    "{sent} Context: {ctx}\n"
                    "Output ONLY the gloss/definition (no extra explanations)."
                )
                stageB_ckpt_dir, _ = train_single_stage(
                    model_name=model_name,
                    train_df=train_df,
                    valid_df=valid_df,
                    stage_output_dir=stageB_dir,
                    prompt_template=stageB_prompt_template,
                    target_col="Meaning_New",
                    ctx_col_for_inputs="Context_New",
                    max_length=256,
                    num_epochs=5,
                    per_device_train_batch_size=8,
                    learning_rate=5e-5,
                    warmup_steps=50,
                    early_stopping_patience=2,
                )

                # Evaluate
                meaning_metrics, context_metrics = evaluate_two_stage(
                    model_name=model_name,
                    test_df=test_df,
                    stageA_ckpt_dir=stageA_ckpt_dir,
                    stageB_ckpt_dir=stageB_ckpt_dir,
                    output_dir=base_model_dir,
                    max_length=256,
                    batch_size=8,
                    num_beams=4,
                    polysemy_aggregation="mean",
                    polysemy_soft_alpha=5.0,
                    # NEW: pass global maps
                    references_map_meaning=references_map_meaning,
                    distinct_count_map_meaning=distinct_count_map_meaning,
                )

                row = {"model_name": model_name}
                row.update({f"meaning_{k}": v for k, v in meaning_metrics.items()})
                row.update({f"context_{k}": v for k, v in context_metrics.items()})
                summary_rows.append(row)

                pd.DataFrame(summary_rows).to_csv(summary_metrics_csv, index=False, encoding="utf-8")
                log.info(f"Finished {model_name}. Meaning metrics: {meaning_metrics}")

            except Exception as e:
                log.exception(f"Failed for model {model_name}: {e}")
                continue

    except Exception as e:
        log.exception(f"Main failed: {e}")

if __name__ == "__main__":
    main()

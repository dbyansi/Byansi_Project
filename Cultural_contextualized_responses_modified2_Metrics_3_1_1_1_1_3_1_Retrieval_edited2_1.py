import os
import sys
import time
import json
import random
import logging
import platform
import socket
from collections import defaultdict
from typing import List, Optional, Dict, Any, Tuple

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

# ---- Logging ----
log_file_path = "NER_Training_Metrics_CCR3_1_1_1_1_20.log"
handlers = [logging.StreamHandler(sys.stdout)]
try:
    handlers.append(logging.FileHandler(log_file_path))
except OSError:
    # disk full or cannot write log file
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
log = logging.getLogger(__name__)

# ---- Determinism / seeds ----
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

# ---- Utility: record environment ----
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


# ---- Data loading & processing ----
def load_data(file_path: str) -> pd.DataFrame:
    log.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, encoding="latin1", low_memory=False)
    log.info(f"Loaded {len(df)} rows")
    return df


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts contexts from <UGE>...</UGE>,
    removes tags from sentences,
    normalizes context tokens (joins spans with spaces),
    preserves Meaning_New per-row without collapsing.
    """
    import re

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

    # Remove tags but keep inner text
    df["processed_sentence"] = df["original_sentence"].apply(
        lambda x: re.sub(r"<UGE>(.*?)</UGE>", r"\1", str(x)).strip()
    )

    # Normalize Meaning_New
    df["Meaning_New"] = df["Meaning_New"].fillna("").astype(str).apply(lambda x: " ".join(x.split()))

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


# ---- Dataset ----
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


# ---- Tokenization helpers (stage-specific, clean) ----
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
    """
    Creates:
      - inputs_encodings: input_ids, attention_mask (or concatenated for causal)
      - labels_encodings: label ids (for seq2seq) OR concatenated-masked labels for causal
    """
    # Build prompts
    prompts = []
    for s, c in zip(texts, ctxs):
        prompts.append(prompt_template.format(sent=s, ctx=c if c is not None else "").strip())

    # Tokenize inputs
    inputs = tokenizer(
        prompts,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_attention_mask=True,
    )

    # Tokenize targets
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
            # concat prompt + target
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

    # seq2seq: separate inputs and labels
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}, labels_input_ids


# ---- Model loading helper ----
def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token

    # Try seq2seq
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
        log.warning(
            f"Could not determine encoder-decoder property for {model_name} from config; trying causal LM fallback."
        )

    # Fallback: causal
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer, True


# ---- Metrics ----
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


def compute_metrics_batch(references: List[str], predictions: List[str]):
    out = {
        "BLEU": [],
        "ROUGE1": [],
        "ROUGE2": [],
        "ROUGEL": [],
        "TER": [],
        "BERTScore": [],
        "EditDistance": [],
        "WER": [],
        "chrF": [],
        "RetriEVAL": [],
    }

    try:
        P, R, F = bert_score.score(predictions, references, lang="en", verbose=False)
        bert_f_list = [f.item() for f in F]
    except Exception:
        bert_f_list = [0.0] * len(predictions)

    for i, (ref, pred) in enumerate(zip(references, predictions)):
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothing_function)
        rouge = calculate_rouge(ref, pred)
        ter = edit_distance(ref.split(), pred.split()) / max(len(ref.split()), 1)
        bert_f = bert_f_list[i] if i < len(bert_f_list) else 0.0
        ed = edit_distance(ref.split(), pred.split())
        w = wer(ref, pred)
        try:
            chrf = corpus_chrf([ref], [pred]).score
        except Exception:
            chrf = 0.0
        retev = compute_retrieval_eval(ref, pred)

        out["BLEU"].append(bleu)
        out["ROUGE1"].append(rouge["rouge1"])
        out["ROUGE2"].append(rouge["rouge2"])  # <-- keep your existing key if you had it; BUT original has "rouge2"
        out["ROUGEL"].append(rouge["rougeL"])
        out["TER"].append(ter)
        out["BERTScore"].append(bert_f)
        out["EditDistance"].append(ed)
        out["WER"].append(w)
        out["chrF"].append(chrf)
        out["RetriEVAL"].append(retev)

    return {k: float(np.mean(v)) if len(v) > 0 else None for k, v in out.items()}
def normalize_key_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = " ".join(s.split())
    return s


def build_polysemy_maps(test_df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Returns:
      references_map[key] = list of distinct Meaning_New values for that key
      distinct_count_map[key] = number of distinct meanings
    """
    references_map = defaultdict(set)
    for _, row in test_df.iterrows():
        key = normalize_key_text(row["processed_sentence"])
        meaning = row["Meaning_New"]
        references_map[key].add(meaning)

    references_map = {k: list(v) for k, v in references_map.items()}
    distinct_count_map = {k: len(v) for k, v in references_map.items()}
    return references_map, distinct_count_map


def compute_metrics_batch_with_polysemy(
    test_df: pd.DataFrame,
    predictions: List[str],
    task_col: str,
    references_map: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Polysemy-aware:
      - For each row i, compute metrics between pred_i and each reference meaning/context
        (all distinct gold values for that key), and take MAX across references.
      - Then return mean across rows.
    """
    # task_col is used only to build "references"; for meaning we use Meaning_New.
    # For context we can reuse same logic but references_map should correspond to that task.

    # we only need polysemy for meaning; context split is optional.
    # We'll implement for any task where references_map is keyed by processed_sentence.

    out = {
        "BLEU": [],
        "ROUGE1": [],
        "ROUGE2": [],
        "ROUGEL": [],
        "TER": [],
        "BERTScore": [],
        "EditDistance": [],
        "WER": [],
        "chrF": [],
        "RetriEVAL": [],
    }

    refs_for_key = references_map

    for i, pred in enumerate(predictions):
        key = normalize_key_text(test_df.iloc[i]["processed_sentence"])
        gold_candidates = refs_for_key.get(key, [])

        # If something goes wrong and there are no references, fallback to empty
        if not gold_candidates:
            gold_candidates = ["" if task_col not in test_df.columns else str(test_df.iloc[i][task_col])]

        # Score against all gold candidates and take best
        # Note: TER/ EditDistance/ WER are "lower is better".
        # We'll keep consistent with metric interpretation by taking MAX for all metrics.
        # You may want to take MIN for TER/ EditDistance/ WER; but to keep behavior consistent with current averaging,
        # we take MAX for metrics that are similarity-like and MAX for others is not ideal.
        # For correctness, we'll treat:
        #   - similarity higher better: BLEU/ROUGE/BERTScore/chrF/RetriEVAL
        #   - distance lower better: TER/EditDistance/WER
        #
        # That gives sensible polysemy handling.
        best = {
            "BLEU": -1e9,
            "ROUGE1": -1e9,
            "ROUGE2": -1e9,
            "ROUGEL": -1e9,
            "TER": 1e18,
            "BERTScore": -1e9,
            "EditDistance": 1e18,
            "WER": 1e18,
            "chrF": -1e9,
            "RetriEVAL": -1e9,
        }

        for ref in gold_candidates:
            bleu = sentence_bleu([ref.split()], str(pred).split(), smoothing_function=smoothing_function)
            rouge = calculate_rouge(ref, pred)
            ter = edit_distance(ref.split(), str(pred).split()) / max(len(ref.split()), 1)

            try:
                # bert_score expects lists; do per-pair for simplicity
                # (costly but keeps change minimal). If need be, we can batch later.
                P, R, F = bert_score.score([str(pred)], [ref], lang="en", verbose=False)
                bert_f = F[0].item()
            except Exception:
                bert_f = 0.0

            ed = edit_distance(ref.split(), str(pred).split())
            w = wer(ref, pred)
            try:
                chrf = corpus_chrf([ref], [str(pred)]).score
            except Exception:
                chrf = 0.0
            retev = compute_retrieval_eval(ref, pred)

            # maximize similarity metrics, minimize distance metrics
            best["BLEU"] = max(best["BLEU"], bleu)
            best["ROUGE1"] = max(best["ROUGE1"], rouge["rouge1"])
            best["ROUGE2"] = max(best["ROUGE2"], rouge["rouge2"])
            best["ROUGEL"] = max(best["ROUGEL"], rouge["rougeL"])
            best["BERTScore"] = max(best["BERTScore"], bert_f)
            best["chrF"] = max(best["chrF"], chrf)
            best["RetriEVAL"] = max(best["RetriEVAL"], retev)

            best["TER"] = min(best["TER"], ter)
            best["EditDistance"] = min(best["EditDistance"], ed)
            best["WER"] = min(best["WER"], w)

        for k in out.keys():
            out[k].append(best[k])

    # mean (None not needed here)
    return {k: float(np.mean(v)) if len(v) > 0 else None for k, v in out.items()}

# ---- Train one stage (context OR meaning) ----
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
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

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


# ---- Inference: predict contexts and meanings ----
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

    prompts = [stage_prompt_template.format(sent=s, ctx=(c if c is not None else "")).strip() for s, c in zip(texts, ctxs)]

    enc = tokenizer(
        prompts,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "early_stopping": True,
    }
    with torch.no_grad():
        n = enc["input_ids"].shape[0]
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            batch_enc = {k: v[start:end] for k, v in enc.items()}
            out_ids = model.generate(**batch_enc, **gen_kwargs)
            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
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


# ---- Two-stage evaluation on test ----
def evaluate_two_stage(
    *,
    model_name: str,
    test_df: pd.DataFrame,
    stageA_ckpt_dir: str,
    stageB_ckpt_dir: str,
    output_dir: str,
    max_length: int,
    batch_size: int = 8,
    num_beams: int = 4,
):
    modelA, tokenizerA, is_causalA = load_from_checkpoint(stageA_ckpt_dir)
    modelB, tokenizerB, is_causalB = load_from_checkpoint(stageB_ckpt_dir)

    test_texts = test_df["processed_sentence"].tolist()
    gold_contexts = test_df["Context_New"].tolist()
    gold_meanings = test_df["Meaning_New"].tolist()

    ctx_empty = [""] * len(test_texts)
    prompt_A = "{sent}"

    pred_contexts = generate_stage_outputs(
        checkpoint_dir=stageA_ckpt_dir,
        model_kind_name="stageA",
        stage_prompt_template=prompt_A,
        texts=test_texts,
        ctxs=ctx_empty,
        tokenizer=tokenizerA,
        model=modelA,
        is_causal=is_causalA,
        max_length=max_length,
        batch_size=batch_size,
        num_beams=num_beams,
    )

    prompt_B = "{sent} Context: {ctx}"
    pred_meanings = generate_stage_outputs(
        checkpoint_dir=stageB_ckpt_dir,
        model_kind_name="stageB",
        stage_prompt_template=prompt_B,
        texts=test_texts,
        ctxs=pred_contexts,
        tokenizer=tokenizerB,
        model=modelB,
        is_causal=is_causalB,
        max_length=max_length,
        batch_size=batch_size,
        num_beams=num_beams,
    )

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Polysemy detection (Meaning)
    # -------------------------
    references_map_meaning, distinct_count_map_meaning = build_polysemy_maps(test_df)

    # build per-row flags
    test_keys = [normalize_key_text(s) for s in test_df["processed_sentence"].tolist()]
    test_df = test_df.copy()
    test_df["key"] = test_keys
    test_df["n_distinct_meanings_for_key"] = test_df["key"].map(distinct_count_map_meaning).fillna(1).astype(int)
    test_df["is_ambiguous_key"] = test_df["n_distinct_meanings_for_key"] >= 2

    # polysemy stats CSV
    poly_stats_rows = []
    for key, refs in references_map_meaning.items():
        poly_stats_rows.append({
            "key": key,
            "n_distinct_meanings": len(refs),
        })
    poly_stats_df = pd.DataFrame(poly_stats_rows)
    poly_stats_path = os.path.join(output_dir, "test_polysemy_stats.csv")
    poly_stats_df.to_csv(poly_stats_path, index=False, encoding="utf-8")

    # Split subsets by ambiguity
    ambiguous_mask = test_df["is_ambiguous_key"].values
    unambiguous_mask = ~ambiguous_mask

    # Overall polysemy-aware meaning metrics: max over refs per key
    aggregated_metrics_meaning_poly = compute_metrics_batch_with_polysemy(
        test_df=test_df,
        predictions=pred_meanings,
        task_col="Meaning_New",
        references_map=references_map_meaning,
    )

    # Ambiguous/unambiguous metrics (still polysemy-aware within each subset)
    test_df_amb = test_df[ambiguous_mask]
    test_df_unamb = test_df[unambiguous_mask]

    pred_meanings_amb = [pred_meanings[i] for i in range(len(pred_meanings)) if ambiguous_mask[i]]
    pred_meanings_unamb = [pred_meanings[i] for i in range(len(pred_meanings)) if unambiguous_mask[i]]

    aggregated_metrics_meaning_amb = compute_metrics_batch_with_polysemy(
        test_df=test_df_amb,
        predictions=pred_meanings_amb,
        task_col="Meaning_New",
        references_map=references_map_meaning,
    ) if len(test_df_amb) > 0 else {k: None for k in aggregated_metrics_meaning_poly.keys()}

    aggregated_metrics_meaning_unamb = compute_metrics_batch_with_polysemy(
        test_df=test_df_unamb,
        predictions=pred_meanings_unamb,
        task_col="Meaning_New",
        references_map=references_map_meaning,
    ) if len(test_df_unamb) > 0 else {k: None for k in aggregated_metrics_meaning_poly.keys()}

    # Context metrics: keep your original behavior (no polysemy-aware scoring unless you want it)
    aggregated_metrics_context = compute_metrics_batch(gold_contexts, pred_contexts)

    # -------------------------
    # Per-example prediction CSV
    # -------------------------
    results_df = pd.DataFrame({
        "model_name": [model_name] * len(test_df),
        "index": test_df.index,
        "processed_sentence": test_df["processed_sentence"].tolist(),
        "gold_context": gold_contexts,
        "predicted_context": pred_contexts,
        "gold_meaning": gold_meanings,
        "predicted_meaning": pred_meanings,
        "key": test_df["key"].tolist(),
        "n_distinct_meanings_for_key": test_df["n_distinct_meanings_for_key"].tolist(),
        "is_ambiguous_key": test_df["is_ambiguous_key"].tolist(),
    })

    results_path = os.path.join(output_dir, "test_predictions_two_stage.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8")

    # -------------------------
    # Meaning metrics CSVs (overall, ambiguous, unambiguous)
    # -------------------------
    meaning_metrics_df = pd.DataFrame(
        [{"model_name": model_name, "task": "meaning", "metric": k, "value": v}
         for k, v in aggregated_metrics_meaning_poly.items()]
    )
    meaning_metrics_path = os.path.join(output_dir, "test_meaning_metrics.csv")
    meaning_metrics_df.to_csv(meaning_metrics_path, index=False, encoding="utf-8")

    meaning_metrics_amb_df = pd.DataFrame(
        [{"model_name": model_name, "task": "meaning_ambiguous", "metric": k, "value": v}
         for k, v in aggregated_metrics_meaning_amb.items()]
    )
    meaning_metrics_amb_path = os.path.join(output_dir, "test_meaning_metrics_ambiguous.csv")
    meaning_metrics_amb_df.to_csv(meaning_metrics_amb_path, index=False, encoding="utf-8")

    meaning_metrics_unamb_df = pd.DataFrame(
        [{"model_name": model_name, "task": "meaning_unambiguous", "metric": k, "value": v}
         for k, v in aggregated_metrics_meaning_unamb.items()]
    )
    meaning_metrics_unamb_path = os.path.join(output_dir, "test_meaning_metrics_unambiguous.csv")
    meaning_metrics_unamb_df.to_csv(meaning_metrics_unamb_path, index=False, encoding="utf-8")

    # -------------------------
    # Context metrics CSV
    # -------------------------
    context_metrics_df = pd.DataFrame(
        [{"model_name": model_name, "task": "context", "metric": k, "value": v}
         for k, v in aggregated_metrics_context.items()]
    )
    context_metrics_path = os.path.join(output_dir, "test_context_metrics.csv")
    context_metrics_df.to_csv(context_metrics_path, index=False, encoding="utf-8")

    # -------------------------
    # Combined summary metrics CSV
    # -------------------------
    combined_metrics_df = pd.concat([meaning_metrics_df, context_metrics_df], ignore_index=True)
    combined_summary_path = os.path.join(output_dir, "test_metrics_summary_two_stage1.csv")
    combined_metrics_df.to_csv(combined_summary_path, index=False, encoding="utf-8")

    # Return: keep your old return signature
    return aggregated_metrics_meaning_poly, aggregated_metrics_context


# ---- MAIN ----
def main():
    try:
        output_dir = "./results/culturally_contextualized_responses_two_stage_2026_06_07"
        os.makedirs(output_dir, exist_ok=True)

        data = load_data("Updated_Training_Dataset_Cultural_Responses_half.csv")
        data = process_data(data)

        data = data[data["processed_sentence"].astype(bool) & data["Meaning_New"].astype(bool)]
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
        valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        log.info(f"Splits : train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

        model_names = [
            'facebook/bart-base',
            't5-base',
            'gpt2',
            'gpt2-medium',
            'gpt2-large',
            'distilgpt2',
            'EleutherAI/gpt-neo-125M',
            'Helsinki-NLP/opus-mt-mul-en',
            'facebook/mbart-large-50',
            'allenai/led-base-16384',
            'google-t5/t5-small',
            'google-t5/t5-large',
            ]


        summary_metrics_csv = os.path.join(output_dir, "all_models_test_metrics_summary_two_stage1.csv")
        summary_rows = []

        for model_name in model_names:
            try:
                log.info(f"Running two-stage training/eval for {model_name}")

                base_model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
                stageA_dir = os.path.join(base_model_dir, "stageA_context")
                stageB_dir = os.path.join(base_model_dir, "stageB_meaning")

                # Stage A
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
                    num_epochs=2,
                    per_device_train_batch_size=8,
                    learning_rate=5e-5,
                    warmup_steps=50,
                    early_stopping_patience=2,
                )

                # Stage B
                stageB_prompt_template = "{sent} Context: {ctx}"
                stageB_ckpt_dir, _ = train_single_stage(
                    model_name=model_name,
                    train_df=train_df,
                    valid_df=valid_df,
                    stage_output_dir=stageB_dir,
                    prompt_template=stageB_prompt_template,
                    target_col="Meaning_New",
                    ctx_col_for_inputs="Context_New",
                    max_length=256,
                    num_epochs=2,
                    per_device_train_batch_size=8,
                    learning_rate=5e-5,
                    warmup_steps=50,
                    early_stopping_patience=2,
                )

                meaning_metrics, context_metrics = evaluate_two_stage(
                    model_name=model_name,
                    test_df=test_df,
                    stageA_ckpt_dir=stageA_ckpt_dir,
                    stageB_ckpt_dir=stageB_ckpt_dir,
                    output_dir=base_model_dir,
                    max_length=256,
                    batch_size=8,
                    num_beams=4,
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

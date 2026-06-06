import os
import re
import pandas as pd

def compute_span_proportions_from_uge(
    df: pd.DataFrame,
    output_dir: str,
    sentence_col: str = "original_sentence",
    meaning_col: str = "Meaning_New",   # used only for polysemy stats
    tag_open: str = "<UGE>",
    tag_close: str = "</UGE>",
    whitespace_norm: bool = True,
    # choose how to handle sentences with multiple <UGE> spans:
    #   "all"   -> count every extracted <UGE>...</UGE> span
    #   "first" -> keep only the first <UGE> span in each sentence row
    #   "discard_multi" -> discard sentence rows that contain 2+ spans
    multi_span_strategy: str = "all",
    # if True, ignore spans whose text becomes empty after normalization
    drop_empty_spans: bool = True,
    save_span_csv: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    # ---- basic checks / prepare ----
    if sentence_col not in df.columns:
        raise ValueError(f"Missing column: {sentence_col}")

    work = df.copy()
    work[sentence_col] = work[sentence_col].fillna("").astype(str)

    has_meaning = (meaning_col in work.columns)
    if has_meaning:
        work[meaning_col] = work[meaning_col].fillna("").astype(str)

    pattern = re.compile(
        rf"{re.escape(tag_open)}(.*?){re.escape(tag_close)}",
        flags=re.DOTALL
    )

    def norm_span_text(x: str) -> str:
        x = "" if x is None else str(x)
        if whitespace_norm:
            x = " ".join(x.strip().split())
        else:
            x = x.strip()
        return x

    span_instances = []  # one row per counted span (or per kept span under your strategy)

    # ---- extract spans ----
    for row_index, sent in zip(work.index.tolist(), work[sentence_col].tolist()):
        spans = pattern.findall(sent)
        if not spans:
            continue

        # apply multi-span strategy
        if multi_span_strategy == "first":
            spans = spans[:1]
        elif multi_span_strategy == "discard_multi":
            if len(spans) >= 2:
                continue
        elif multi_span_strategy == "all":
            pass
        else:
            raise ValueError("multi_span_strategy must be one of: 'all', 'first', 'discard_multi'")

        meaning_val = work.loc[row_index, meaning_col] if has_meaning else None

        # add span rows
        for span_id, span in enumerate(spans):
            span_text = norm_span_text(span)

            if drop_empty_spans and (span_text == ""):
                continue

            word_count = len(span_text.split())
            is_multiword = (word_count >= 2)

            span_instances.append({
                "row_index": row_index,
                "span_id_in_row": span_id,
                "span_text": span_text,
                "word_count": word_count,
                "is_multiword": int(is_multiword),
                "meaning": meaning_val,
            })

    # ---- compute proportions ----
    if len(span_instances) == 0:
        out = {
            "num_span_instances": 0,

            "num_multiword_spans": 0,
            "proportion_multiword_spans": 0.0,

            "num_singleword_spans": 0,  # <-- INCLUDED
            "proportion_singleword_spans": 0.0,

            "num_unique_span_types": 0,
            "num_multiword_span_types": 0,
            "proportion_multiword_span_types": 0.0,

            # (optional but included for completeness)
            "num_singleword_span_types": 0,
            "proportion_singleword_span_types": 0.0,
        }

        if has_meaning:
            out["num_span_types_with_ge2_meanings"] = 0
            out["proportion_span_types_with_ge2_meanings"] = 0.0

    else:
        spans_df = pd.DataFrame(span_instances)

        num_instances = len(spans_df)
        num_multiword = int(spans_df["is_multiword"].sum())
        num_singleword = int(num_instances - num_multiword)

        out = {
            "num_span_instances": int(num_instances),

            "num_multiword_spans": num_multiword,
            "proportion_multiword_spans": num_multiword / max(num_instances, 1),

            "num_singleword_spans": num_singleword,  # <-- INCLUDED
            "proportion_singleword_spans": num_singleword / max(num_instances, 1),
        }

        # span "types" = unique span_text
        type_df = spans_df.groupby("span_text").agg(
            type_word_count=("word_count", "max"),
            num_occurrences=("row_index", "count"),
        ).reset_index()

        num_unique_types = len(type_df)
        num_multiword_types = int((type_df["type_word_count"] >= 2).sum())
        num_singleword_types = int(num_unique_types - num_multiword_types)

        out.update({
            "num_unique_span_types": int(num_unique_types),

            "num_multiword_span_types": num_multiword_types,
            "proportion_multiword_span_types": num_multiword_types / max(num_unique_types, 1),

            "num_singleword_span_types": num_singleword_types,
            "proportion_singleword_span_types": num_singleword_types / max(num_unique_types, 1),
        })

        # polysemy (if meaning exists): span types that appear with >=2 distinct meaning strings
        if has_meaning:
            poly = spans_df.groupby("span_text")["meaning"].nunique(dropna=True)
            num_poly_types = int((poly >= 2).sum())
            out["num_span_types_with_ge2_meanings"] = num_poly_types
            out["proportion_span_types_with_ge2_meanings"] = num_poly_types / max(num_unique_types, 1)

        # ---- save span-level CSV (optional) ----
        if save_span_csv:
            spans_csv_path = os.path.join(output_dir, "extracted_uge_spans_spanlevel.csv")
            spans_df.to_csv(spans_csv_path, index=False, encoding="utf-8")

    # ---- save summary CSV ----
    summary_path = os.path.join(output_dir, "span_proportions_summary.csv")
    pd.DataFrame([out]).to_csv(summary_path, index=False, encoding="utf-8")

    return out


# ------------------ EXAMPLE USAGE ------------------
if __name__ == "__main__":
    input_path = "Updated_Training_Dataset_Cultural_Responses.csv"
    output_dir = "./results/span_stats_example1"

    df = pd.read_csv(input_path, encoding="latin1", low_memory=False)

    proportions = compute_span_proportions_from_uge(
        df=df,
        output_dir=output_dir,
        sentence_col="sentence",
        meaning_col="Meaning_New",
        multi_span_strategy="all",
        save_span_csv=True,
    )

    print("Span proportions:", proportions)

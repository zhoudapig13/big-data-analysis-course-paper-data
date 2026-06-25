"""Clean CFPB complaint records and construct analysis features."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "cfpb_combined_raw.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

CHATGPT_RELEASE_DATE = pd.Timestamp("2022-11-30")

LEGAL_TERMS = [
    "fcra",
    "fdcpa",
    "cfpb",
    "15 u.s.c",
    "15 usc",
    "fair credit reporting act",
    "truth in lending",
    "equal credit opportunity",
    "violation",
    "statute",
    "pursuant",
    "section",
    "rights",
    "written consent",
    "identity theft",
]
EVIDENCE_TERMS = [
    "attached",
    "attachment",
    "document",
    "proof",
    "evidence",
    "statement",
    "screenshot",
    "letter",
    "police report",
    "ftc report",
    "affidavit",
]
URGENCY_TERMS = [
    "urgent",
    "immediately",
    "as soon as possible",
    "asap",
    "please help",
    "cannot",
    "unable",
    "emergency",
    "overdue",
    "late fee",
]
HARDSHIP_TERMS = [
    "homeless",
    "eviction",
    "rent",
    "job",
    "employment",
    "medical",
    "illness",
    "disability",
    "veteran",
    "servicemember",
    "financial hardship",
    "depressed",
]
TEMPLATE_PHRASES = [
    "in accordance with",
    "federally protected consumer",
    "without my written consent",
    "fair credit reporting act",
    "pursuant to section",
    "under 15 u.s.c",
    "15 u.s.c 1681",
    "15 usc 1681",
    "violation of my rights",
    "doctrine of estoppel",
]
POLITENESS_TERMS = ["please", "thank you", "appreciate", "kindly"]
ANGER_TERMS = [
    "fraud",
    "illegal",
    "deceptive",
    "abusive",
    "unfair",
    "lie",
    "lying",
    "refuse",
    "denied",
    "harass",
]


def snake_case(name: str) -> str:
    value = name.strip().lower()
    value = value.replace("?", "")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: snake_case(str(c)) for c in df.columns})
    aliases = {
        "date_received": ["date_received"],
        "product": ["product"],
        "sub_product": ["sub_product"],
        "issue": ["issue"],
        "sub_issue": ["sub_issue"],
        "consumer_narrative": [
            "consumer_complaint_narrative",
            "consumer_narrative",
            "complaint_what_happened",
        ],
        "company_public_response": ["company_public_response"],
        "company": ["company"],
        "state": ["state"],
        "zip_code": ["zip_code", "zip"],
        "tags": ["tags"],
        "consumer_consent": [
            "consumer_consent_provided",
            "consumer_consent",
        ],
        "submitted_via": ["submitted_via"],
        "date_sent_to_company": ["date_sent_to_company"],
        "company_response": [
            "company_response_to_consumer",
            "company_response",
        ],
        "timely_response": ["timely_response", "timely"],
        "consumer_disputed": ["consumer_disputed"],
        "complaint_id": ["complaint_id"],
        "raw_source": ["raw_source"],
    }
    out = pd.DataFrame(index=df.index)
    for canonical, candidates in aliases.items():
        found = [col for col in candidates if col in df.columns]
        if not found:
            out[canonical] = pd.NA
            continue
        pieces = []
        for col in found:
            data = df.loc[:, df.columns == col]
            if data.shape[1] == 1:
                pieces.append(data.iloc[:, 0])
            else:
                pieces.append(data.bfill(axis=1).iloc[:, 0])
        out[canonical] = pd.concat(pieces, axis=1).bfill(axis=1).iloc[:, 0]
    return out


def parse_dates(series: pd.Series) -> pd.Series:
    # pandas 2.x supports format="mixed"; fallback is kept for older runtimes.
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def parse_binary_yes(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.map(
        {
            "yes": 1,
            "y": 1,
            "true": 1,
            "1": 1,
            "no": 0,
            "n": 0,
            "false": 0,
            "0": 0,
        }
    ).astype("float")


def count_terms(text: str, terms: list[str]) -> int:
    text_l = text.lower()
    return int(sum(text_l.count(term) for term in terms))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", text.lower())


def build_text_features(narrative: pd.Series) -> pd.DataFrame:
    texts = narrative.fillna("").astype(str)
    rows = []
    for text in texts:
        tokens = tokenize(text)
        token_count = len(tokens)
        sentence_count = max(1, len(re.findall(r"[.!?]+", text))) if text else 0
        letters = re.findall(r"[A-Za-z]", text)
        upper_letters = re.findall(r"[A-Z]", text)
        unique_tokens = len(set(tokens))
        word_lengths = [len(tok) for tok in tokens if tok.isalpha()]
        rows.append(
            {
                "n_chars": len(text),
                "n_words": token_count,
                "n_sentences": sentence_count,
                "avg_word_len": float(np.mean(word_lengths)) if word_lengths else 0.0,
                "avg_sentence_len": token_count / sentence_count
                if sentence_count
                else 0.0,
                "lexical_diversity": unique_tokens / token_count
                if token_count
                else 0.0,
                "uppercase_ratio": len(upper_letters) / len(letters)
                if letters
                else 0.0,
                "digit_count": len(re.findall(r"\d", text)),
                "money_amount_count": len(
                    re.findall(r"\{\$[0-9,]+(?:\.[0-9]+)?\}|\$[0-9,]+", text)
                ),
                "redaction_count": len(re.findall(r"\bX{2,}\b", text)),
                "question_count": text.count("?"),
                "exclamation_count": text.count("!"),
                "legal_term_count": count_terms(text, LEGAL_TERMS),
                "evidence_term_count": count_terms(text, EVIDENCE_TERMS),
                "urgency_term_count": count_terms(text, URGENCY_TERMS),
                "hardship_term_count": count_terms(text, HARDSHIP_TERMS),
                "template_phrase_count": count_terms(text, TEMPLATE_PHRASES),
                "politeness_count": count_terms(text, POLITENESS_TERMS),
                "anger_term_count": count_terms(text, ANGER_TERMS),
            }
        )
    return pd.DataFrame(rows, index=narrative.index)


def normalize_product(product: pd.Series) -> pd.Series:
    p = product.fillna("").astype(str).str.lower()
    out = pd.Series("Other", index=product.index, dtype="string")
    out[p.str.contains("credit reporting|consumer reports", regex=True)] = (
        "Credit reporting"
    )
    out[p.str.contains("debt collection", regex=False)] = "Debt collection"
    out[p.str.contains("credit card|prepaid", regex=True)] = "Credit card/prepaid"
    out[p.str.contains("checking|savings|bank account", regex=True)] = "Bank account"
    out[p.str.contains("mortgage", regex=False)] = "Mortgage"
    out[p.str.contains("student loan", regex=False)] = "Student loan"
    out[p.str.contains("vehicle loan|lease", regex=True)] = "Vehicle loan/lease"
    out[p.str.contains("payday|personal loan|consumer loan", regex=True)] = (
        "Consumer loan"
    )
    out[p.str.contains("money transfer|virtual currency", regex=True)] = (
        "Money transfer"
    )
    return out


def classify_response(response: pd.Series) -> pd.DataFrame:
    r = response.fillna("").astype(str).str.lower().str.strip()
    non_monetary = r.str.contains("non-monetary relief", regex=False)
    monetary = r.str.contains("monetary relief", regex=False) & ~non_monetary
    explanation = r.str.contains("explanation", regex=False)
    closed = r.str.contains("closed", regex=False)
    outcome = pd.Series("other", index=response.index, dtype="string")
    outcome[explanation] = "explanation"
    outcome[non_monetary] = "non_monetary_relief"
    outcome[monetary] = "monetary_relief"
    return pd.DataFrame(
        {
            "response_outcome": outcome,
            "monetary_relief": monetary.astype(int),
            "non_monetary_relief": non_monetary.astype(int),
            "any_relief": (monetary | non_monetary).astype(int),
            "closed_with_explanation": explanation.astype(int),
            "closed_response": closed.astype(int),
        },
        index=response.index,
    )


def feature_dictionary() -> pd.DataFrame:
    rows = [
        ("post_chatgpt", "1 if date_received is on/after 2022-11-30."),
        ("any_relief", "1 for monetary or non-monetary relief responses."),
        ("monetary_relief", "1 for closed with monetary relief."),
        ("non_monetary_relief", "1 for closed with non-monetary relief."),
        ("timely_binary", "1 if CFPB marks company response as timely."),
        ("n_words", "Number of lexical tokens in the complaint narrative."),
        ("avg_sentence_len", "Average tokens per sentence."),
        ("lexical_diversity", "Unique token ratio."),
        ("legal_term_count", "Count of legal/regulatory cue terms."),
        ("evidence_term_count", "Count of evidence/document cue terms."),
        ("urgency_term_count", "Count of urgency cue terms."),
        ("hardship_term_count", "Count of hardship cue terms."),
        ("template_phrase_count", "Count of repeated legal/template phrases."),
        ("product_group", "Coarsened product category for modeling."),
    ]
    return pd.DataFrame(rows, columns=["variable", "definition"])


def clean_features(input_path: Path, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(input_path, low_memory=False)
    df = standardize_columns(raw)

    before = len(df)
    df["date_received"] = parse_dates(df["date_received"])
    df["date_sent_to_company"] = parse_dates(df["date_sent_to_company"])
    df["complaint_id"] = df["complaint_id"].astype("string")
    df = df.drop_duplicates(subset=["complaint_id"], keep="first")

    for col in [
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "consumer_narrative",
        "company_public_response",
        "company",
        "state",
        "zip_code",
        "tags",
        "consumer_consent",
        "submitted_via",
        "company_response",
        "consumer_disputed",
        "raw_source",
    ]:
        df[col] = df[col].astype("string")

    df["consumer_narrative"] = df["consumer_narrative"].fillna("")
    df["has_narrative"] = df["consumer_narrative"].str.strip().ne("").astype(int)
    df["product_group"] = normalize_product(df["product"])
    df["timely_binary"] = parse_binary_yes(df["timely_response"])
    df["consumer_disputed_binary"] = parse_binary_yes(df["consumer_disputed"])
    df["days_to_company"] = (
        df["date_sent_to_company"] - df["date_received"]
    ).dt.days

    df["complaint_year"] = df["date_received"].dt.year
    df["complaint_month"] = df["date_received"].dt.to_period("M").astype("string")
    df["complaint_quarter"] = df["date_received"].dt.to_period("Q").astype("string")
    df["post_chatgpt"] = (df["date_received"] >= CHATGPT_RELEASE_DATE).astype(int)

    response_features = classify_response(df["company_response"])
    text_features = build_text_features(df["consumer_narrative"])
    df = pd.concat([df, response_features, text_features], axis=1)

    modeling = df[
        (df["has_narrative"] == 1)
        & df["date_received"].notna()
        & df["company_response"].notna()
    ].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / "cfpb_clean_features.csv"
    model_path = output_dir / "cfpb_modeling_dataset.csv"
    dict_path = output_dir / "feature_dictionary.csv"
    meta_path = output_dir / "cleaning_metadata.json"

    df.to_csv(clean_path, index=False, encoding="utf-8")
    modeling.to_csv(model_path, index=False, encoding="utf-8")
    feature_dictionary().to_csv(dict_path, index=False, encoding="utf-8")

    metadata = {
        "input_path": str(input_path),
        "raw_rows": int(before),
        "deduplicated_rows": int(len(df)),
        "modeling_rows": int(len(modeling)),
        "date_min": str(modeling["date_received"].min().date())
        if not modeling.empty
        else None,
        "date_max": str(modeling["date_received"].max().date())
        if not modeling.empty
        else None,
        "clean_path": str(clean_path),
        "modeling_path": str(model_path),
        "feature_dictionary_path": str(dict_path),
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return modeling, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAW_PATH))
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modeling, metadata = clean_features(Path(args.input), Path(args.output_dir))
    print(f"Cleaned modeling rows: {len(modeling):,}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

"""Generate profiling tables, topic model, and baseline prediction models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "cfpb_modeling_dataset.csv"
TABLE_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "outputs" / "models"

NUMERIC_FEATURES = [
    "n_chars",
    "n_words",
    "n_sentences",
    "avg_word_len",
    "avg_sentence_len",
    "lexical_diversity",
    "uppercase_ratio",
    "digit_count",
    "money_amount_count",
    "redaction_count",
    "question_count",
    "exclamation_count",
    "legal_term_count",
    "evidence_term_count",
    "urgency_term_count",
    "hardship_term_count",
    "template_phrase_count",
    "politeness_count",
    "anger_term_count",
    "post_chatgpt",
]
CATEGORICAL_FEATURES = ["product_group", "submitted_via", "state"]
TEXT_FEATURE = "consumer_narrative"
CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS).union(
    {
        "xx",
        "xxx",
        "xxxx",
        "xxxxx",
        "xxxxxx",
        "xxxxxxx",
        "xxxxxxxx",
        "xxxxxxxxx",
        "xxxxxxxxxx",
        "00",
        "000",
        "0000",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
        "account",
        "accounts",
        "consumer",
        "complaint",
        "company",
        "credit",
    }
)
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b"


def safe_rate(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def write_basic_tables(df: pd.DataFrame, table_dir: Path) -> dict:
    table_dir.mkdir(parents=True, exist_ok=True)
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

    profile = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "date_min": str(df["date_received"].min().date()),
        "date_max": str(df["date_received"].max().date()),
        "unique_products": int(df["product"].nunique(dropna=True)),
        "unique_issues": int(df["issue"].nunique(dropna=True)),
        "unique_companies": int(df["company"].nunique(dropna=True)),
        "unique_states": int(df["state"].nunique(dropna=True)),
        "post_chatgpt_share": safe_rate(df["post_chatgpt"]),
        "any_relief_rate": safe_rate(df["any_relief"]),
        "monetary_relief_rate": safe_rate(df["monetary_relief"]),
        "non_monetary_relief_rate": safe_rate(df["non_monetary_relief"]),
        "timely_response_rate": safe_rate(df["timely_binary"]),
        "median_words": float(df["n_words"].median()),
        "mean_words": float(df["n_words"].mean()),
    }
    with (table_dir / "dataset_profile.json").open("w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)

    top_specs = [
        ("product_group", "top_product_groups.csv"),
        ("product", "top_products.csv"),
        ("issue", "top_issues.csv"),
        ("company", "top_companies.csv"),
        ("state", "top_states.csv"),
        ("response_outcome", "response_outcomes.csv"),
    ]
    for col, name in top_specs:
        (
            df[col]
            .fillna("Missing")
            .value_counts()
            .rename_axis(col)
            .reset_index(name="n")
            .assign(share=lambda x: x["n"] / len(df))
            .head(50)
            .to_csv(table_dir / name, index=False, encoding="utf-8")
        )

    monthly = (
        df.groupby("complaint_month")
        .agg(
            n=("complaint_id", "size"),
            any_relief_rate=("any_relief", "mean"),
            monetary_relief_rate=("monetary_relief", "mean"),
            non_monetary_relief_rate=("non_monetary_relief", "mean"),
            timely_rate=("timely_binary", "mean"),
            mean_words=("n_words", "mean"),
            template_phrase_rate=("template_phrase_count", "mean"),
        )
        .reset_index()
        .sort_values("complaint_month")
    )
    monthly.to_csv(table_dir / "monthly_profile.csv", index=False, encoding="utf-8")

    product_response = (
        df.groupby("product_group")
        .agg(
            n=("complaint_id", "size"),
            any_relief_rate=("any_relief", "mean"),
            monetary_relief_rate=("monetary_relief", "mean"),
            non_monetary_relief_rate=("non_monetary_relief", "mean"),
            timely_rate=("timely_binary", "mean"),
            mean_words=("n_words", "mean"),
            legal_terms=("legal_term_count", "mean"),
            template_phrases=("template_phrase_count", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    product_response.to_csv(
        table_dir / "product_response_rates.csv", index=False, encoding="utf-8"
    )

    issue_response = (
        df.groupby("issue")
        .agg(
            n=("complaint_id", "size"),
            any_relief_rate=("any_relief", "mean"),
            timely_rate=("timely_binary", "mean"),
            mean_words=("n_words", "mean"),
            template_phrases=("template_phrase_count", "mean"),
        )
        .query("n >= 50")
        .reset_index()
        .sort_values("n", ascending=False)
        .head(50)
    )
    issue_response.to_csv(
        table_dir / "issue_response_rates_top50.csv", index=False, encoding="utf-8"
    )

    pre_post_features = [
        "any_relief",
        "monetary_relief",
        "non_monetary_relief",
        "timely_binary",
        "n_words",
        "avg_sentence_len",
        "lexical_diversity",
        "legal_term_count",
        "evidence_term_count",
        "urgency_term_count",
        "hardship_term_count",
        "template_phrase_count",
    ]
    pre_post = (
        df.groupby("post_chatgpt")[pre_post_features]
        .mean(numeric_only=True)
        .reset_index()
    )
    pre_post["period"] = np.where(
        pre_post["post_chatgpt"].eq(1), "post_2022_11_30", "pre_2022_11_30"
    )
    pre_post.to_csv(
        table_dir / "pre_post_chatgpt_proxy_summary.csv",
        index=False,
        encoding="utf-8",
    )

    corr_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + ["any_relief"]
    correlations = (
        df[corr_cols]
        .apply(pd.to_numeric, errors="coerce")
        .corr(numeric_only=True)["any_relief"]
        .drop("any_relief", errors="ignore")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .reset_index()
    )
    correlations.columns = ["feature", "correlation_with_any_relief"]
    correlations.to_csv(
        table_dir / "feature_correlations_any_relief.csv",
        index=False,
        encoding="utf-8",
    )

    figure_blueprint = pd.DataFrame(
        [
            (
                "Figure 1",
                "Data pipeline mechanism",
                "Raw complaints -> cleaning -> text features -> topics -> response models.",
            ),
            (
                "Figure 2",
                "Monthly complaint and relief trends",
                "Line chart or small multiples using monthly_profile.csv.",
            ),
            (
                "Figure 3",
                "Product-risk heatmap",
                "Product group by response outcome/rate from product_response_rates.csv.",
            ),
            (
                "Figure 4",
                "Topic mechanism map",
                "Topic prevalence and response rates from topic_response_rates.csv.",
            ),
            (
                "Figure 5",
                "Model explanation",
                "Top positive/negative logistic features or later SHAP values.",
            ),
            (
                "Figure 6",
                "Quasi-causal mechanism",
                "Post-ChatGPT proxy, complaint expression, and company response path.",
            ),
        ],
        columns=["figure_id", "draft_title", "data_or_design"],
    )
    figure_blueprint.to_csv(
        table_dir / "figure_blueprint.csv", index=False, encoding="utf-8"
    )
    return profile


def run_topic_model(df: pd.DataFrame, table_dir: Path, model_dir: Path) -> pd.DataFrame:
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(
        stop_words=list(CUSTOM_STOP_WORDS),
        max_features=8000,
        min_df=5,
        max_df=0.90,
        ngram_range=(1, 2),
        token_pattern=TOKEN_PATTERN,
    )
    x = vectorizer.fit_transform(df[TEXT_FEATURE].fillna(""))
    n_topics = 8 if len(df) >= 2000 else 5
    nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=300)
    weights = nmf.fit_transform(x)
    terms = np.array(vectorizer.get_feature_names_out())

    keyword_rows = []
    for topic_idx, component in enumerate(nmf.components_):
        top_idx = component.argsort()[::-1][:15]
        keyword_rows.append(
            {
                "topic": int(topic_idx),
                "keywords": ", ".join(terms[top_idx]),
                "top_weight": float(component[top_idx[0]]),
            }
        )
    pd.DataFrame(keyword_rows).to_csv(
        table_dir / "topic_keywords_nmf.csv", index=False, encoding="utf-8"
    )

    out = df.copy()
    out["dominant_topic"] = weights.argmax(axis=1)
    out["dominant_topic_weight"] = weights.max(axis=1)
    topic_profile = (
        out.groupby("dominant_topic")
        .agg(
            n=("complaint_id", "size"),
            share=("complaint_id", lambda s: len(s) / len(out)),
            any_relief_rate=("any_relief", "mean"),
            monetary_relief_rate=("monetary_relief", "mean"),
            non_monetary_relief_rate=("non_monetary_relief", "mean"),
            timely_rate=("timely_binary", "mean"),
            mean_words=("n_words", "mean"),
            template_phrases=("template_phrase_count", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    topic_profile.to_csv(
        table_dir / "topic_response_rates.csv", index=False, encoding="utf-8"
    )
    out.to_csv(
        ROOT / "data" / "processed" / "cfpb_modeling_dataset_with_topics.csv",
        index=False,
        encoding="utf-8",
    )
    return out


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if name == "text":
            names.extend(transformer.get_feature_names_out())
        elif name == "cat":
            names.extend(transformer.get_feature_names_out(columns))
        elif name == "num":
            names.extend(columns)
    return [str(n) for n in names]


def run_binary_model(
    df: pd.DataFrame,
    target: str,
    table_dir: Path,
    model_dir: Path,
) -> dict:
    usable = df.dropna(subset=[target]).copy()
    usable[target] = usable[target].astype(int)
    class_counts = usable[target].value_counts()
    if len(class_counts) < 2 or class_counts.min() < 30:
        note = {
            "target": target,
            "status": "skipped",
            "reason": "Target has fewer than two usable classes or a minority class below 30.",
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        }
        with (model_dir / f"model_metrics_{target}.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(note, fh, indent=2)
        return note

    features = [TEXT_FEATURE] + CATEGORICAL_FEATURES + NUMERIC_FEATURES
    for col in CATEGORICAL_FEATURES:
        usable[col] = usable[col].fillna("Missing").astype(str)
    for col in NUMERIC_FEATURES:
        usable[col] = pd.to_numeric(usable[col], errors="coerce").fillna(0)

    train, test = train_test_split(
        usable,
        test_size=0.25,
        random_state=42,
        stratify=usable[target],
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    stop_words=list(CUSTOM_STOP_WORDS),
                    max_features=6000,
                    min_df=5,
                    max_df=0.90,
                    ngram_range=(1, 2),
                    token_pattern=TOKEN_PATTERN,
                ),
                TEXT_FEATURE,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(with_mean=False), NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train[features], train[target])
    proba = model.predict_proba(test[features])[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "target": target,
        "status": "fit",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "positive_rate_train": float(train[target].mean()),
        "positive_rate_test": float(test[target].mean()),
        "accuracy": float(accuracy_score(test[target], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test[target], pred)),
        "f1": float(f1_score(test[target], pred)),
        "roc_auc": float(roc_auc_score(test[target], proba)),
        "average_precision": float(average_precision_score(test[target], proba)),
        "classification_report": classification_report(
            test[target], pred, output_dict=True
        ),
    }
    with (model_dir / f"model_metrics_{target}.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(metrics, fh, indent=2)

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    names = transformed_feature_names(prep)
    coefs = clf.coef_[0]
    feature_table = pd.DataFrame({"feature": names, "coef": coefs})
    feature_table["abs_coef"] = feature_table["coef"].abs()
    top_features = pd.concat(
        [
            feature_table.sort_values("coef", ascending=False).head(40).assign(
                direction="positive"
            ),
            feature_table.sort_values("coef", ascending=True).head(40).assign(
                direction="negative"
            ),
        ],
        ignore_index=True,
    )
    top_features.to_csv(
        table_dir / f"model_top_features_{target}.csv",
        index=False,
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DATA_PATH))
    parser.add_argument("--table-dir", default=str(TABLE_DIR))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_dir = Path(args.table_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    profile = write_basic_tables(df, table_dir)
    df_topics = run_topic_model(df, table_dir, model_dir)
    relief_metrics = run_binary_model(df_topics, "any_relief", table_dir, model_dir)
    timely_metrics = run_binary_model(
        df_topics.dropna(subset=["timely_binary"]), "timely_binary", table_dir, model_dir
    )

    run_summary = {
        "profile": profile,
        "models": {
            "any_relief": {
                k: v
                for k, v in relief_metrics.items()
                if k not in {"classification_report"}
            },
            "timely_binary": {
                k: v
                for k, v in timely_metrics.items()
                if k not in {"classification_report"}
            },
        },
        "key_outputs": {
            "dataset_profile": str(table_dir / "dataset_profile.json"),
            "monthly_profile": str(table_dir / "monthly_profile.csv"),
            "topic_keywords": str(table_dir / "topic_keywords_nmf.csv"),
            "topic_response_rates": str(table_dir / "topic_response_rates.csv"),
            "model_features_any_relief": str(
                table_dir / "model_top_features_any_relief.csv"
            ),
        },
    }
    with (table_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(run_summary, fh, indent=2)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, classification_report,
    roc_curve
)

# -----------------------------
# Config
# -----------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

STUDENT_INFO = os.path.join(RAW_DIR, "studentInfo.csv")
STUDENT_VLE = os.path.join(RAW_DIR, "studentVle.csv")
VLE = os.path.join(RAW_DIR, "vle.csv")

N_WEEKS = 7
MAX_DAY = N_WEEKS * 7  # 49
CHUNKSIZE = 2_000_000  # adjust if needed

KEYS = ["code_module", "code_presentation", "id_student"]

RANDOM_STATE = 42


def safe_read_csv(path: str, usecols=None, dtype=None, chunksize=None):
    """Read CSV with utf-8 fallback."""
    try:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=chunksize, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=chunksize, encoding="latin1", low_memory=False)


def build_features_first_7_weeks(student_vle_path: str, vle_path: str) -> pd.DataFrame:
    """
    Build per (module, presentation, student) features for first 7 weeks:
    - total_clicks
    - clicks by activity_type (pivot)
    """
    print("Loading vle (id_site -> activity_type)...")
    vle_df = safe_read_csv(
        vle_path,
        usecols=["id_site", "activity_type"],
        dtype={"id_site": "int32", "activity_type": "string"},
        chunksize=None
    )
    vle_map = pd.Series(vle_df["activity_type"].values, index=vle_df["id_site"].values)
    # Some id_site may not map; keep as "unknown"
    print(f"vle rows: {len(vle_df):,}")

    # studentVle columns sometimes differ; handle both sum_click / sum_clicks
    usecols = ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    dtypes = {
        "code_module": "category",
        "code_presentation": "category",
        "id_student": "int32",
        "id_site": "int32",
        "date": "int16",
        "sum_click": "int32",
    }

    print("Streaming studentVle in chunks and aggregating (first 7 weeks)...")
    acc = None  # aggregated Series with MultiIndex: KEYS + activity_type

    # Try reading with sum_click; if fails, fall back to sum_clicks
    try:
        iterator = safe_read_csv(student_vle_path, usecols=usecols, dtype=dtypes, chunksize=CHUNKSIZE)
    except ValueError:
        usecols_alt = ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_clicks"]
        dtypes_alt = dtypes.copy()
        dtypes_alt.pop("sum_click")
        dtypes_alt["sum_clicks"] = "int32"
        iterator = safe_read_csv(student_vle_path, usecols=usecols_alt, dtype=dtypes_alt, chunksize=CHUNKSIZE)

    processed_rows = 0
    kept_rows = 0

    for i, chunk in enumerate(iterator, start=1):
        processed_rows += len(chunk)

        # unify column name
        if "sum_clicks" in chunk.columns and "sum_click" not in chunk.columns:
            chunk = chunk.rename(columns={"sum_clicks": "sum_click"})

        # filter first 7 weeks (0~49 days)
        chunk = chunk[(chunk["date"] >= 0) & (chunk["date"] <= MAX_DAY)]
        kept_rows += len(chunk)

        if len(chunk) == 0:
            continue

        # map id_site -> activity_type
        chunk["activity_type"] = chunk["id_site"].map(vle_map).astype("string")
        chunk["activity_type"] = chunk["activity_type"].fillna("unknown")

        # group by KEYS + activity_type
        grp = chunk.groupby(KEYS + ["activity_type"], observed=True)["sum_click"].sum()

        if acc is None:
            acc = grp
        else:
            acc = acc.add(grp, fill_value=0)

        if i % 5 == 0:
            print(f"  chunks processed: {i}, rows read: {processed_rows:,}, kept (0-49d): {kept_rows:,}")

    if acc is None:
        raise RuntimeError("No data left after filtering first 7 weeks. Check date range / input files.")

    agg_df = acc.reset_index(name="clicks")

    # Pivot to wide features by activity_type
    feat = agg_df.pivot_table(
        index=KEYS,
        columns="activity_type",
        values="clicks",
        aggfunc="sum",
        fill_value=0
    )

    # total clicks
    feat["total_clicks"] = feat.sum(axis=1)

    # flatten columns
    feat.columns = [f"clicks__{c}" if c != "total_clicks" else "total_clicks" for c in feat.columns]
    feat = feat.reset_index()

    print(f"Feature table shape: {feat.shape} (rows x cols)")
    return feat


def load_labels(student_info_path: str) -> pd.DataFrame:
    print("Loading studentInfo (labels)...")
    info = safe_read_csv(
        student_info_path,
        usecols=KEYS + ["final_result"],
        dtype={"code_module": "category", "code_presentation": "category", "id_student": "int32", "final_result": "string"},
        chunksize=None
    )
    # label: Withdrawn = 1 else 0
    info["y"] = (info["final_result"] == "Withdrawn").astype(int)
    return info[KEYS + ["y"]]


def plot_confusion(cm: np.ndarray, title: str, outpath: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Non-Withdrawn", "Withdrawn"])
    plt.yticks([0, 1], ["Non-Withdrawn", "Withdrawn"])
    for (r, c), v in np.ndenumerate(cm):
        plt.text(c, r, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    # Build X
    X_feat = build_features_first_7_weeks(STUDENT_VLE, VLE)

    # Load y and merge
    y_df = load_labels(STUDENT_INFO)
    data = X_feat.merge(y_df, on=KEYS, how="inner")

    # Separate X/y
    y = data["y"].values
    X = data.drop(columns=KEYS + ["y"])

    # Basic cleanup
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # -----------------------------
    # Model 1: Logistic Regression
    # -----------------------------
    logreg = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False)),  # sparse-like safe
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE
        ))
    ])

    # -----------------------------
    # Model 2: Random Forest
    # -----------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    models = {
        "LogReg": logreg,
        "RandomForest": rf
    }

    metrics_lines = []
    roc_data = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)

        # Probabilities for AUC / ROC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # fallback
            y_proba = model.decision_function(X_test)

        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)

        metrics_lines.append(f"[{name}] AUC={auc:.4f}, F1={f1:.4f}")
        metrics_lines.append(f"[{name}] Confusion Matrix:\n{cm}\n")
        metrics_lines.append(f"[{name}] Classification Report:\n{classification_report(y_test, y_pred, digits=4)}\n")

        # Save confusion matrix plot
        cm_path = os.path.join(OUT_DIR, f"supervised_cm_{name}.png")
        plot_confusion(cm, f"Confusion Matrix ({name})", cm_path)

        # Save ROC points
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)

    # Save ROC curve figure (both models)
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (First {N_WEEKS} Weeks Features)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "supervised_roc_logreg_rf.png"), dpi=200)
    plt.close()

    # Feature importance (RF)
    try:
        rf_model = models["RandomForest"]
        importances = rf_model.feature_importances_
        feat_names = X.columns.to_list()
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(15)

        plt.figure(figsize=(8, 6))
        plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
        plt.xlabel("Importance")
        plt.title("Random Forest Feature Importance (Top 15)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "supervised_rf_feature_importance_top15.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"Skip RF feature importance due to: {e}")

    # Save metrics to txt
    metrics_path = os.path.join(OUT_DIR, "supervised_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metrics_lines))

    print("\nDone!")
    print(f"Saved figures + metrics to: {OUT_DIR}")
    print(f"Metrics file: {metrics_path}")


if __name__ == "__main__":
    main()

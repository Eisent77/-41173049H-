import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -----------------------------
# Settings
# -----------------------------
DATA_DIR = r"C:\Users\fish9\Desktop\Project\data\raw"  # 你的 csv 若在同一資料夾就用 "."
OUTPUT_DIR = "outputs_unsupervised"
FIRST_N_WEEKS = 7           # 前7週
DAYS_PER_WEEK = 7
MAX_DAY = FIRST_N_WEEKS * DAYS_PER_WEEK  # 49
RANDOM_STATE = 42

# K 候選範圍
K_MIN, K_MAX = 2, 10

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_csvs(data_dir: str):
    student_vle = pd.read_csv(os.path.join(data_dir, "studentVle.csv"))
    student_info = pd.read_csv(os.path.join(data_dir, "studentInfo.csv"))
    vle = pd.read_csv(os.path.join(data_dir, "vle.csv"))
    return student_vle, student_info, vle

def build_first7week_features(student_vle: pd.DataFrame, vle: pd.DataFrame):
    """
    產出每筆樣本 (code_module, code_presentation, id_student) 的
    - sum_click_total_7w
    - activity_type_*_7w (依 vle.csv 的 activity_type)
    """
    # 只留前 7 週
    sv = student_vle.copy()
    sv = sv[(sv["date"] >= 0) & (sv["date"] < MAX_DAY)]

    # 合併 activity_type（需要 vle_id）
    # studentVle: (code_module, code_presentation, id_student, id_site, date, sum_click)
    # vle: (id_site, code_module, code_presentation, activity_type, ...)
    sv = sv.merge(
        vle[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"],
        how="left"
    )

    # 1) 總 clicks（前7週）
    key_cols = ["code_module", "code_presentation", "id_student"]
    total_7w = sv.groupby(key_cols)["sum_click"].sum().reset_index()
    total_7w = total_7w.rename(columns={"sum_click": "sum_click_total_7w"})

    # 2) 各 activity_type clicks（前7週）
    pivot = (
        sv.pivot_table(
            index=key_cols,
            columns="activity_type",
            values="sum_click",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    # 欄位名整理
    act_cols = [c for c in pivot.columns if c not in key_cols]
    pivot = pivot.rename(columns={c: f"act_{c}_7w" for c in act_cols})

    # 合併成一張 feature table
    feat = total_7w.merge(pivot, on=key_cols, how="left")
    feat = feat.fillna(0)

    return feat

def pick_k_by_elbow(inertias, ks):
    # 這裡不做自動找拐點（避免誤判），你用圖決定即可
    return None

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(OUTPUT_DIR)
    print("Loading CSVs...")
    student_vle, student_info, vle = load_csvs(DATA_DIR)

    print("Building features (first 7 weeks)...")
    feat = build_first7week_features(student_vle, vle)

    # 把 label 留著「之後對照用」（不拿來訓練）
    # studentInfo: final_result = 'Withdrawn' / 'Pass' / ...
    label_df = student_info[["code_module", "code_presentation", "id_student", "final_result"]].copy()
    df = feat.merge(label_df, on=["code_module", "code_presentation", "id_student"], how="left")

    # 只取 feature columns
    key_cols = ["code_module", "code_presentation", "id_student"]
    feature_cols = [c for c in df.columns if c not in key_cols + ["final_result"]]
    X = df[feature_cols].astype(float)

    # Standardize（KMeans 很吃尺度）
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # -----------------------------
    # 1) Elbow (Inertia)
    # -----------------------------
    ks = list(range(K_MIN, K_MAX + 1))
    inertias = []
    sils = []

    print("Running KMeans for K range...")
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)

        # silhouette 需要 k>=2，且資料不能全同群
        try:
            s = silhouette_score(Xs, labels)
        except Exception:
            s = np.nan
        sils.append(s)

    # Elbow plot
    plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Method (Inertia) - First 7 Weeks Features")
    plt.xlabel("K (number of clusters)")
    plt.ylabel("Inertia")
    elbow_path = os.path.join(OUTPUT_DIR, "unsup_elbow_inertia.png")
    plt.savefig(elbow_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Silhouette plot（可用來輔助挑 K）
    plt.figure()
    plt.plot(ks, sils, marker="o")
    plt.title("Silhouette Score - First 7 Weeks Features")
    plt.xlabel("K (number of clusters)")
    plt.ylabel("Silhouette score")
    sil_path = os.path.join(OUTPUT_DIR, "unsup_silhouette.png")
    plt.savefig(sil_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {elbow_path}")
    print(f"Saved: {sil_path}")

    # -----------------------------
    # 2) Choose a K (你先手動改這個)
    # -----------------------------
    K_CHOSEN = 4  # <= 你先用圖決定，再改這個數字
    print(f"Fitting final KMeans with K={K_CHOSEN} ...")
    km_final = KMeans(n_clusters=K_CHOSEN, random_state=RANDOM_STATE, n_init=10)
    cluster_id = km_final.fit_predict(Xs)
    df["cluster"] = cluster_id

    # -----------------------------
    # 3) PCA 2D scatter
    # -----------------------------
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(Xs)
    df["pca1"] = Z[:, 0]
    df["pca2"] = Z[:, 1]

    plt.figure()
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        plt.scatter(sub["pca1"], sub["pca2"], s=8, label=f"Cluster {c}", alpha=0.7)
    plt.title(f"PCA 2D Scatter (KMeans K={K_CHOSEN})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    pca_path = os.path.join(OUTPUT_DIR, f"unsup_pca_scatter_k{K_CHOSEN}.png")
    plt.savefig(pca_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pca_path}")

    # -----------------------------
    # 4) Cluster summary (size + mean features + withdrawn rate for explanation)
    # -----------------------------
    # Withdrawn indicator（只用來解釋，不拿來訓練）
    df["is_withdrawn"] = (df["final_result"].astype(str).str.lower() == "withdrawn").astype(int)

    summary = df.groupby("cluster").agg(
        n_samples=("id_student", "count"),
        withdrawn_rate=("is_withdrawn", "mean"),
        mean_total_clicks=("sum_click_total_7w", "mean"),
        median_total_clicks=("sum_click_total_7w", "median")
    ).reset_index()

    # 每群 top activity features（平均值最高的前3個）
    act_cols = [c for c in feature_cols if c.startswith("act_")]
    top_feats = []
    for c in sorted(df["cluster"].unique()):
        means = df[df["cluster"] == c][act_cols].mean().sort_values(ascending=False)
        top3 = list(means.head(3).index)
        top_feats.append(", ".join(top3))
    summary["top3_activity_features"] = top_feats

    summary_path = os.path.join(OUTPUT_DIR, f"unsup_cluster_summary_k{K_CHOSEN}.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {summary_path}")

    # 也把帶 cluster 的資料存起來
    out_df_path = os.path.join(OUTPUT_DIR, f"unsup_assignments_k{K_CHOSEN}.csv")
    df[key_cols + ["final_result", "cluster", "pca1", "pca2", "sum_click_total_7w"]].to_csv(
        out_df_path, index=False, encoding="utf-8-sig"
    )
    print(f"Saved: {out_df_path}")

    print("\nDone! Check outputs in:", os.path.abspath(OUTPUT_DIR))
    print("Next: open elbow/silhouette plots, decide K, then re-run with K_CHOSEN updated.")

if __name__ == "__main__":
    main()

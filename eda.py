import os
import pandas as pd
import matplotlib.pyplot as plt

# ====== settings ======
ROOT = r"C:\Users\fish9\Desktop\Project"
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

N_WEEKS = 7
MAX_DAY = N_WEEKS * 7  # 49 days

PATH_STUDENT_INFO = os.path.join(RAW, "studentInfo.csv")
PATH_STUDENT_VLE  = os.path.join(RAW, "studentVle.csv")
PATH_VLE          = os.path.join(RAW, "vle.csv")

# ====== load ======
print("Loading CSVs...")
studentInfo = pd.read_csv(PATH_STUDENT_INFO)
studentVle  = pd.read_csv(PATH_STUDENT_VLE)
vle         = pd.read_csv(PATH_VLE)

# ====== label ======
# y = 1 if Withdrawn else 0
studentInfo["y_withdrawn"] = (studentInfo["final_result"] == "Withdrawn").astype(int)

# keep only essential columns for merging
info = studentInfo[["code_module", "code_presentation", "id_student", "final_result", "y_withdrawn"]].copy()

# ====== filter time window (first 7 weeks) ======
# OULAD date is relative to course start; keep 0..49 days
vle_f = studentVle[(studentVle["date"] >= 0) & (studentVle["date"] <= MAX_DAY)].copy()

# ====== EDA 1: label distribution ======
label_counts = info["y_withdrawn"].value_counts().sort_index()
# index 0 = non-withdrawn, 1 = withdrawn
plt.figure()
plt.bar(["Non-Withdrawn", "Withdrawn"], [label_counts.get(0, 0), label_counts.get(1, 0)])
plt.title("Withdrawal Distribution (Label)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "eda_label_distribution.png"), dpi=200)
plt.close()

# ====== feature 1: total clicks in first N weeks ======
# aggregate clicks per (module, presentation, student)
agg_total = (
    vle_f.groupby(["code_module", "code_presentation", "id_student"], as_index=False)["sum_click"]
    .sum()
    .rename(columns={"sum_click": f"clicks_sum_w1_w{N_WEEKS}"})
)

df_total = info.merge(agg_total, on=["code_module", "code_presentation", "id_student"], how="left")
df_total[f"clicks_sum_w1_w{N_WEEKS}"] = df_total[f"clicks_sum_w1_w{N_WEEKS}"].fillna(0)

# ====== EDA 2: clicks distribution by label ======
# simple boxplot (two groups)
withdrawn = df_total[df_total["y_withdrawn"] == 1][f"clicks_sum_w1_w{N_WEEKS}"]
non_withd = df_total[df_total["y_withdrawn"] == 0][f"clicks_sum_w1_w{N_WEEKS}"]

plt.figure()
plt.boxplot([non_withd, withdrawn], labels=["Non-Withdrawn", "Withdrawn"], showfliers=False)
plt.title(f"Total Clicks in First {N_WEEKS} Weeks (0–{MAX_DAY} days)")
plt.ylabel("Total clicks (sum_click)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, f"eda_clicks_box_w1_w{N_WEEKS}.png"), dpi=200)
plt.close()

# ====== EDA 3: weekly trend (mean clicks per week by label) ======
# add week index
vle_f["week"] = (vle_f["date"] // 7 + 1).astype(int)
vle_f = vle_f[vle_f["week"].between(1, N_WEEKS)]

# total clicks per student per week
stu_week = (
    vle_f.groupby(["code_module", "code_presentation", "id_student", "week"], as_index=False)["sum_click"]
    .sum()
)

# attach label
stu_week = stu_week.merge(info[["code_module", "code_presentation", "id_student", "y_withdrawn"]],
                          on=["code_module", "code_presentation", "id_student"], how="left")

trend = stu_week.groupby(["y_withdrawn", "week"], as_index=False)["sum_click"].mean()

# plot two lines
plt.figure()
for y_val, name in [(0, "Non-Withdrawn"), (1, "Withdrawn")]:
    sub = trend[trend["y_withdrawn"] == y_val].sort_values("week")
    plt.plot(sub["week"], sub["sum_click"], marker="o", label=name)

plt.title(f"Weekly Mean Clicks Trend (First {N_WEEKS} Weeks)")
plt.xlabel("Week")
plt.ylabel("Mean weekly clicks")
plt.xticks(range(1, N_WEEKS + 1))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, f"eda_weekly_trend_w1_w{N_WEEKS}.png"), dpi=200)
plt.close()

# ====== (Optional) EDA 4: activity_type top comparison ======
# join to activity_type
vle_small = vle[["id_site", "activity_type"]].copy()
vle_join = vle_f.merge(vle_small, on="id_site", how="left")

atype = (
    vle_join.groupby(["code_module", "code_presentation", "id_student", "activity_type"], as_index=False)["sum_click"]
    .sum()
)

atype = atype.merge(info[["code_module", "code_presentation", "id_student", "y_withdrawn"]],
                    on=["code_module", "code_presentation", "id_student"], how="left")

atype_mean = atype.groupby(["y_withdrawn", "activity_type"], as_index=False)["sum_click"].mean()

# pick top 8 activity_types by overall mean
overall = atype_mean.groupby("activity_type", as_index=False)["sum_click"].mean()
top_types = overall.sort_values("sum_click", ascending=False).head(8)["activity_type"].tolist()

atype_plot = atype_mean[atype_mean["activity_type"].isin(top_types)].copy()

# pivot for plotting
pivot = atype_plot.pivot(index="activity_type", columns="y_withdrawn", values="sum_click").fillna(0)
pivot = pivot.sort_values(by=0, ascending=False)

plt.figure(figsize=(10, 5))
x = range(len(pivot.index))
plt.bar([i - 0.2 for i in x], pivot.get(0, 0), width=0.4, label="Non-Withdrawn")
plt.bar([i + 0.2 for i in x], pivot.get(1, 0), width=0.4, label="Withdrawn")
plt.xticks(list(x), pivot.index, rotation=30, ha="right")
plt.title("Mean Clicks by activity_type (Top 8, First 7 Weeks)")
plt.ylabel("Mean clicks")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, f"eda_activity_type_top8_w1_w{N_WEEKS}.png"), dpi=200)
plt.close()

print("Done! Figures saved to:", OUT)

"""
Online Shopper Purchase Propensity Model
----------------------------------------
Predicts whether a website session ends in a purchase (Revenue = True) using
the UCI Online Shoppers Purchasing Intention dataset, then shows how a marketing
team could use the scores to prioritize high-intent visitors.

Run:
    pip install pandas scikit-learn matplotlib
    python purchase_propensity.py                         # downloads data from UCI
    python purchase_propensity.py path/to/file.csv        # or use a local copy

Outputs are saved to ./outputs (charts + results.txt).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00468/online_shoppers_intention.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
log = []


def say(text=""):
    print(text)
    log.append(str(text))


# ---------------------------------------------------------------- 1. Load + clean
source = sys.argv[1] if len(sys.argv) > 1 else DATA_URL
df = pd.read_csv(source)
say(f"Loaded {len(df):,} sessions, {df.shape[1]} columns")
say(f"Missing values: {int(df.isna().sum().sum())}")
say(f"Duplicate rows: {int(df.duplicated().sum())} (dropping them)")
df = df.drop_duplicates().reset_index(drop=True)

df["Revenue"] = df["Revenue"].astype(int)
df["Weekend"] = df["Weekend"].astype(int)
say(f"Overall conversion rate: {df['Revenue'].mean():.1%}\n")

# ---------------------------------------------------------------- 2. Explore
by_visitor = df.groupby("VisitorType")["Revenue"].mean().sort_values()
by_month = df.groupby("Month")["Revenue"].agg(["mean", "size"])
month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
by_month = by_month.reindex([m for m in month_order if m in by_month.index])

say("Conversion rate by visitor type:")
say(by_visitor.map("{:.1%}".format).to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
by_visitor.plot(kind="barh", ax=axes[0], color="#2a6f97")
axes[0].set_title("Conversion rate by visitor type")
axes[0].set_xlabel("Share of sessions ending in purchase")
axes[1].bar(by_month.index, by_month["mean"], color="#2a6f97")
axes[1].set_title("Conversion rate by month")
axes[1].set_ylabel("Share of sessions ending in purchase")
plt.tight_layout()
plt.savefig(OUT / "conversion_by_segment.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 3. Model prep
y = df["Revenue"]
X = df.drop(columns="Revenue")
categorical = ["Month", "VisitorType", "OperatingSystems", "Browser",
               "Region", "TrafficType"]
numeric = [c for c in X.columns if c not in categorical]

prep = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1),
}

# ---------------------------------------------------------------- 4. Train + evaluate
say("\nBaseline: always predicting 'no purchase' is "
    f"{1 - y_test.mean():.1%} accurate but finds 0% of buyers, "
    "so ROC AUC and recall matter more than accuracy here.\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fitted, scores = {}, {}
for name, model in models.items():
    pipe = Pipeline([("prep", prep), ("model", model)])
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fitted[name], scores[name] = pipe, proba

    say(f"=== {name} ===")
    say(f"5-fold CV ROC AUC (train): {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
    say(f"Test ROC AUC:              {roc_auc_score(y_test, proba):.3f}")
    say(f"Test PR AUC:               {average_precision_score(y_test, proba):.3f}")
    say("Confusion matrix [[TN FP] [FN TP]]:")
    say(confusion_matrix(y_test, pred))
    say(classification_report(y_test, pred, target_names=["No purchase", "Purchase"]))

best = max(scores, key=lambda n: roc_auc_score(y_test, scores[n]))
say(f"Best model on test ROC AUC: {best}\n")

# ---------------------------------------------------------------- 5. What drives purchases?
rf_pipe = fitted["Random Forest"]
feature_names = rf_pipe.named_steps["prep"].get_feature_names_out()
importances = pd.Series(
    rf_pipe.named_steps["model"].feature_importances_, index=feature_names
).sort_values(ascending=False)
importances.index = [i.replace("num__", "").replace("cat__", "") for i in importances.index]

say("Top 8 features (Random Forest importance):")
say(importances.head(8).round(3).to_string())

plt.figure(figsize=(7, 4.5))
importances.head(10)[::-1].plot(kind="barh", color="#2a6f97")
plt.title("Top 10 drivers of purchase (Random Forest)")
plt.xlabel("Feature importance")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 6. Marketing use case: lift
scored = pd.DataFrame({"actual": y_test.values, "score": scores[best]})
scored["decile"] = pd.qcut(scored["score"].rank(method="first", ascending=False),
                           10, labels=range(1, 11))
lift = scored.groupby("decile", observed=True)["actual"].agg(["sum", "mean"])
lift["cum_share_of_buyers"] = lift["sum"].cumsum() / lift["sum"].sum()
top20 = lift.loc[2, "cum_share_of_buyers"]
say("\nMarketing view: if we only targeted the top 20% of sessions by score,")
say(f"we would reach {top20:.0%} of all buyers "
    f"(vs. 20% with random targeting, a {top20 / 0.2:.1f}x lift).")

plt.figure(figsize=(6, 4))
plt.plot([0] + list(range(10, 101, 10)),
         [0] + list(lift["cum_share_of_buyers"] * 100), marker="o",
         label="Model", color="#2a6f97")
plt.plot([0, 100], [0, 100], "--", color="gray", label="Random targeting")
plt.xlabel("% of sessions targeted (highest score first)")
plt.ylabel("% of buyers reached")
plt.title("Cumulative gains")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "cumulative_gains.png", dpi=150)
plt.close()

(OUT / "results.txt").write_text("\n".join(log))
say(f"\nSaved charts and results.txt to {OUT.resolve()}")

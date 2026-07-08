# ============================================================
# PART A - Traditional Machine Learning: Spam Classification
# Dataset: email.csv (Category, Message)
# Models: Logistic Regression & Naive Bayes with TF-IDF
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
import re
import string

# ── Create plots folder if not present ──────────────────────
os.makedirs("plots", exist_ok=True)


# ============================================================
# 1. LOAD & CLEAN DATA
# ============================================================
print("=" * 55)
print("STEP 1: Loading and Cleaning Dataset")
print("=" * 55)

df = pd.read_csv("data/email.csv")
print(f"Raw shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Remove junk rows (non ham/spam labels)
df = df[df["Category"].isin(["ham", "spam"])].copy()
df.reset_index(drop=True, inplace=True)
print(f"Cleaned shape: {df.shape}")

# Rename for convenience
df.rename(columns={"Category": "label", "Message": "text"}, inplace=True)

# Encode labels: spam=1, ham=0
df["label_enc"] = df["label"].map({"spam": 1, "ham": 0})

print(f"\nClass Distribution:")
print(df["label"].value_counts())
print(f"\nSpam %: {df['label_enc'].mean()*100:.2f}%")


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 55)
print("STEP 2: Exploratory Data Analysis")
print("=" * 55)

# Add message length feature for EDA
df["msg_length"] = df["text"].apply(len)
print(df.groupby("label")["msg_length"].describe())

# Plot 1: Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
counts = df["label"].value_counts()
axes[0].bar(counts.index, counts.values, color=["steelblue", "tomato"], edgecolor="black")
axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Label")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")

# Message length distribution
for label, color in zip(["ham", "spam"], ["steelblue", "tomato"]):
    axes[1].hist(df[df["label"] == label]["msg_length"],
                 bins=50, alpha=0.6, label=label, color=color)
axes[1].set_title("Message Length Distribution", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Message Length (characters)")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/eda_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plots/eda_distribution.png")


# ============================================================
# 3. TEXT PREPROCESSING
# ============================================================
print("\n" + "=" * 55)
print("STEP 3: Text Preprocessing")
print("=" * 55)

def preprocess_text(text):
    """
    Preprocessing steps:
    1. Lowercase
    2. Remove URLs
    3. Remove punctuation and digits
    4. Strip extra whitespace
    Justification: Lowercasing reduces vocabulary size.
    URLs and punctuation add noise without semantic value.
    Digits in spam messages are often irrelevant (e.g., phone numbers).
    We intentionally skip stopword removal so TF-IDF can still
    down-weight common terms naturally via IDF.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)                      # remove digits
    text = re.sub(r"\s+", " ", text).strip()             # remove extra spaces
    return text

df["text_clean"] = df["text"].apply(preprocess_text)
print("Sample before:", df["text"][2])
print("Sample after :", df["text_clean"][2])


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================
print("\n" + "=" * 55)
print("STEP 4: Train/Test Split")
print("=" * 55)

"""
Split strategy: 80% train, 20% test with stratify=True.
Justification: Stratified split ensures both sets maintain
the same class ratio (~13% spam), preventing leakage of
imbalance into evaluation.
"""
X_train, X_test, y_train, y_test = train_test_split(
    df["text_clean"], df["label_enc"],
    test_size=0.2, random_state=42, stratify=df["label_enc"]
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Train spam %: {y_train.mean()*100:.2f}% | Test spam %: {y_test.mean()*100:.2f}%")


# ============================================================
# 5. TF-IDF VECTORIZATION
# ============================================================
print("\n" + "=" * 55)
print("STEP 5: TF-IDF Vectorization")
print("=" * 55)

"""
TF-IDF parameters:
- max_features=10000: top 10k terms keep vocab manageable
- ngram_range=(1,2): unigrams + bigrams capture phrases like
  "free entry", "click here" common in spam
- min_df=2: ignore terms appearing in only 1 document
- sublinear_tf=True: apply log normalization to term frequency
  to reduce impact of highly repeated terms
"""
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"TF-IDF matrix shape (train): {X_train_tfidf.shape}")


# ============================================================
# 6. TRAIN MODELS & EVALUATE
# ============================================================
print("\n" + "=" * 55)
print("STEP 6: Training and Evaluating Models")
print("=" * 55)

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """Train model, record time, return metrics dict."""
    start = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start

    y_pred = model.predict(X_te)

    metrics = {
        "Model": name,
        "Accuracy":  accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred),
        "Recall":    recall_score(y_te, y_pred),
        "F1-Score":  f1_score(y_te, y_pred),
        "Train Time (s)": round(train_time, 4),
        "y_pred": y_pred
    }

    print(f"\n── {name} ──")
    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1-Score : {metrics['F1-Score']:.4f}")
    print(f"Train Time: {metrics['Train Time (s)']} s")
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=["Ham", "Spam"]))

    return metrics

# Logistic Regression
lr_model = LogisticRegression(
    C=1.0,           # regularization strength
    max_iter=1000,
    solver="lbfgs",
    random_state=42
)
lr_results = evaluate_model(
    "Logistic Regression", lr_model,
    X_train_tfidf, y_train,
    X_test_tfidf,  y_test
)

# Naive Bayes
nb_model = MultinomialNB(alpha=0.1)   # alpha: Laplace smoothing
nb_results = evaluate_model(
    "Naive Bayes", nb_model,
    X_train_tfidf, y_train,
    X_test_tfidf,  y_test
)


# ============================================================
# 7. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 55)
print("STEP 7: Visualizations")
print("=" * 55)

models     = ["Logistic Regression", "Naive Bayes"]
metrics    = ["Accuracy", "Precision", "Recall", "F1-Score"]
lr_vals    = [lr_results[m] for m in metrics]
nb_vals    = [nb_results[m] for m in metrics]

# ── Plot 2: Metric Comparison Bar Chart ──
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, lr_vals, width, label="Logistic Regression",
               color="steelblue", edgecolor="black")
bars2 = ax.bar(x + width/2, nb_vals, width, label="Naive Bayes",
               color="tomato", edgecolor="black")

ax.set_ylim(0.85, 1.02)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison (Part A)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("plots/metric_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plots/metric_comparison.png")

# ── Plot 3: Confusion Matrices ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, results, title in zip(
    axes,
    [lr_results, nb_results],
    ["Logistic Regression", "Naive Bayes"]
):
    cm = confusion_matrix(y_test, results["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"])
    ax.set_title(f"Confusion Matrix: {title}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.savefig("plots/confusion_matrices_partA.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plots/confusion_matrices_partA.png")

# ── Plot 4: Training Time Comparison ──
fig, ax = plt.subplots(figsize=(6, 4))
times  = [lr_results["Train Time (s)"], nb_results["Train Time (s)"]]
colors = ["steelblue", "tomato"]
bars   = ax.bar(models, times, color=colors, edgecolor="black", width=0.4)
ax.set_title("Training Time Comparison (Part A)", fontsize=13, fontweight="bold")
ax.set_ylabel("Time (seconds)")
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
            f"{t:.4f}s", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("plots/training_time_partA.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plots/training_time_partA.png")


# ============================================================
# 8. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)

summary_df = pd.DataFrame([
    {k: v for k, v in lr_results.items() if k != "y_pred"},
    {k: v for k, v in nb_results.items() if k != "y_pred"},
])
summary_df.set_index("Model", inplace=True)
print(summary_df.to_string())
print("\nPart A Complete! All plots saved in the plots/ folder.")
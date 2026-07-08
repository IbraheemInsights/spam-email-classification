# ============================================================
# PART B - Transformer-Based Model: Spam Classification
# Model: DistilBERT (Hugging Face) with Custom PyTorch Loop
# No Trainer API used
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import re
import string

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

# ── Plots folder ─────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

# ── Device: use GPU if available, else CPU ───────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# 1. LOAD & CLEAN DATA
# ============================================================
print("\n" + "=" * 55)
print("STEP 1: Loading and Cleaning Dataset")
print("=" * 55)

df = pd.read_csv("data/email.csv")
df = df[df["Category"].isin(["ham", "spam"])].copy()
df.reset_index(drop=True, inplace=True)
df.rename(columns={"Category": "label", "Message": "text"}, inplace=True)
df["label_enc"] = df["label"].map({"spam": 1, "ham": 0})
print(f"Dataset shape: {df.shape}")
print(df["label"].value_counts())


# ============================================================
# 2. LIGHT PREPROCESSING
# ============================================================
print("\n" + "=" * 55)
print("STEP 2: Preprocessing")
print("=" * 55)

"""
For DistilBERT we do minimal preprocessing.
The tokenizer handles subword splitting and special tokens.
We only remove URLs and extra whitespace — heavy cleaning
(stopword removal, stemming) would destroy contextual signals
that BERT-based models rely on.
"""
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text_clean"] = df["text"].apply(preprocess_text)
print("Preprocessing done.")


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================
print("\n" + "=" * 55)
print("STEP 3: Train/Test Split (80/20 stratified)")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    df["text_clean"].tolist(),
    df["label_enc"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label_enc"]
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")


# ============================================================
# 4. TOKENIZER & DATASET
# ============================================================
print("\n" + "=" * 55)
print("STEP 4: Loading DistilBERT Tokenizer")
print("=" * 55)

"""
MAX_LEN=128: Most SMS messages are short; 128 tokens covers
~95% of messages while keeping memory usage low on CPU.
Longer sequences increase compute cost quadratically.
"""
MAX_LEN   = 128
BATCH_SIZE = 16    # Small batch for CPU compatibility
EPOCHS     = 3
LR         = 2e-5  # Standard fine-tuning LR for BERT models

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer loaded.")

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = SpamDataset(X_train, y_train, tokenizer, MAX_LEN)
test_dataset  = SpamDataset(X_test,  y_test,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")


# ============================================================
# 5. MODEL SETUP
# ============================================================
print("\n" + "=" * 55)
print("STEP 5: Loading DistilBERT Model")
print("=" * 55)

"""
Layer freezing strategy:
We freeze the first 4 transformer layers and only fine-tune
the last 2 layers + classifier head.
Justification: Early layers capture general linguistic features
(syntax, grammar) that transfer well. Fine-tuning only upper
layers reduces overfitting on small datasets and speeds up
training on CPU significantly.
"""
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Freeze embedding layer
for param in model.distilbert.embeddings.parameters():
    param.requires_grad = False

# Freeze first 4 of 6 transformer layers
for i in range(4):
    for param in model.distilbert.transformer.layer[i].parameters():
        param.requires_grad = False

model.to(device)

total_params   = sum(p.numel() for p in model.parameters())
trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params   : {total_params:,}")
print(f"Trainable params: {trainable:,}")
print(f"Frozen params  : {total_params - trainable:,}")


# ============================================================
# 6. CUSTOM PYTORCH TRAINING LOOP
# ============================================================
print("\n" + "=" * 55)
print("STEP 6: Training DistilBERT")
print("=" * 55)

"""
Optimizer: AdamW with weight decay for regularization.
Loss: CrossEntropyLoss (built into DistilBertForSequenceClassification).
No scheduler used to keep the loop simple and interpretable.
"""
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=0.01   # L2 regularization
)

train_losses = []
train_accuracies = []
epoch_times = []

total_train_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()

    total_loss    = 0
    correct       = 0
    total_samples = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 40)

    for batch_idx, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item()
        preds          = torch.argmax(logits, dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total_samples
    epoch_time = time.time() - epoch_start

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    epoch_times.append(epoch_time)

    print(f"  Epoch {epoch+1} Summary → Loss: {avg_loss:.4f} | "
          f"Acc: {accuracy:.4f} | Time: {epoch_time:.1f}s")

total_train_time = time.time() - total_train_start
print(f"\nTotal Training Time: {total_train_time:.1f}s")


# ============================================================
# 7. EVALUATION
# ============================================================
print("\n" + "=" * 55)
print("STEP 7: Evaluating on Test Set")
print("=" * 55)

model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall    = recall_score(all_labels, all_preds)
f1        = f1_score(all_labels, all_preds)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Ham", "Spam"]))


# ============================================================
# 8. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 55)
print("STEP 8: Saving Visualizations")
print("=" * 55)

# ── Plot 1: Training Loss Curve ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, EPOCHS+1), train_losses, marker="o",
             color="steelblue", linewidth=2, markersize=8)
axes[0].set_title("Training Loss per Epoch", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_xticks(range(1, EPOCHS+1))
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, EPOCHS+1), train_accuracies, marker="s",
             color="tomato", linewidth=2, markersize=8)
axes[1].set_title("Training Accuracy per Epoch", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_xticks(range(1, EPOCHS+1))
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/training_curves_partB.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plots/training_curves_partB.png")

# ── Plot 2: Confusion Matrix ──
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"])
ax.set_title("Confusion Matrix: DistilBERT", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig("plots/confusion_matrix_partB.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plots/confusion_matrix_partB.png")

# ── Plot 3: All Models Metric Comparison ──
# Part A results (from part_a.py output)
lr_scores = {"Accuracy": 0.963, "Precision": 1.000, "Recall": 0.725, "F1-Score": 0.840}
nb_scores = {"Accuracy": 0.985, "Precision": 1.000, "Recall": 0.886, "F1-Score": 0.940}
db_scores = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}

metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(metrics_list))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width, [lr_scores[m] for m in metrics_list], width,
               label="Logistic Regression", color="steelblue", edgecolor="black")
bars2 = ax.bar(x,         [nb_scores[m] for m in metrics_list], width,
               label="Naive Bayes",         color="tomato",    edgecolor="black")
bars3 = ax.bar(x + width, [db_scores[m] for m in metrics_list], width,
               label="DistilBERT",          color="seagreen",  edgecolor="black")

ax.set_ylim(0.65, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(metrics_list, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("All Models Performance Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("plots/all_models_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plots/all_models_comparison.png")

# ── Plot 4: Training Time Comparison ──
model_names = ["Logistic Regression", "Naive Bayes", "DistilBERT"]
times_list  = [0.1073, 0.0120, total_train_time]
colors      = ["steelblue", "tomato", "seagreen"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(model_names, times_list, color=colors, edgecolor="black", width=0.4)
ax.set_title("Training Time Comparison — All Models", fontsize=13, fontweight="bold")
ax.set_ylabel("Time (seconds)")
for bar, t in zip(bars, times_list):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{t:.1f}s", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/training_time_all.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plots/training_time_all.png")


# ============================================================
# 9. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 55)
print("FINAL SUMMARY — ALL MODELS")
print("=" * 55)

summary = pd.DataFrame([
    {"Model": "Logistic Regression", **lr_scores, "Train Time (s)": 0.1073},
    {"Model": "Naive Bayes",         **nb_scores, "Train Time (s)": 0.0120},
    {"Model": "DistilBERT",          **db_scores, "Train Time (s)": round(total_train_time, 1)},
])
summary.set_index("Model", inplace=True)
print(summary.to_string())
print("\nPart B Complete! All plots saved in the plots/ folder.")
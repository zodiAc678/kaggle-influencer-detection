# Influencer or Observer: Predicting Social Roles from a Single Tweet

Data challenge from the **Deep Learning** course at **École Polytechnique** (December 2025), hosted on Kaggle.

The task: given a single tweet (text + metadata of its author), predict whether it was written by an **Influencer** (label `1`) or an **Observer** (label `0`). The dataset contains 154,914 training tweets and 103,380 test tweets, evaluated on **accuracy**.

**Final leaderboard score: 0.843 accuracy.**

## Results

| Approach | Accuracy |
|---|---|
| TF-IDF + Logistic Regression (baseline) | ~0.63 (validation) |
| Hybrid XGBoost: 194 handcrafted features + Twitter-RoBERTa embeddings | 0.839 (leaderboard) |
| **Final pipeline** (features + RoBERTa/PCA + TF-IDF/SVD + K-Means + GBDT ensemble + pseudo-labelling) | **0.843 (leaderboard)** |

## Final pipeline

```
tweet + user metadata
        │
        ├── ~50 behavioural features ─────────────────────────────┐
        │   (log follower/friend ratios, engagement, account age,  │
        │    profile completeness, device, bio keywords…)          │
        │                                                          │
        └── text = "tweet [SEP] user description"                  │
              ├── Twitter-RoBERTa [CLS] (768d) → PCA (50d) ────────┤
              │         └── K-Means (12 clusters, one-hot) ────────┤
              └── TF-IDF 1–2 grams (10k) → TruncatedSVD (100d) ────┤
                                                                   ▼
                              LightGBM + XGBoost + CatBoost (weighted average,
                              weights via scipy.optimize, threshold tuned on validation)
                                                                   │
                              safe pseudo-labelling (p > 0.95 or p < 0.05)
                              → retrain ensemble → submission.csv
```

Key ideas:

- **Metadata matters more than text.** Network asymmetry (followers vs friends, listed count) is the strongest signal; text alone (TF-IDF, fine-tuned BERTweet) plateaus well below.
- **Text still adds value** once combined with features: contextual embeddings from [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), compressed with PCA to limit noise and overfitting.
- **User groups**: K-Means clusters on the embeddings act as a coarse "type of account" feature.
- **Regularised GBDT ensemble** with early stopping, then **pseudo-labelling** on high-confidence test predictions (~8–12% of the test set) for a small extra gain.

More details in the [report](report/report.pdf).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md                 # how to get the data (not versioned)
├── notebooks/
│   ├── final_solution.ipynb      # final pipeline → submission.csv
│   └── exploration/              # earlier approaches, in chronological order
│       └── README.md
└── report/
    └── report.pdf                # 5-page write-up of the approach
```

## Reproducing the results

1. Create an environment (Python 3.11 was used) and install dependencies:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
   For GPU support, install the PyTorch build matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).
2. Download `train.jsonl` and `kaggle_test.jsonl` and place them in `data/` (see [data/README.md](data/README.md)).
3. Run all cells of [`notebooks/final_solution.ipynb`](notebooks/final_solution.ipynb) in order.
4. The predictions are written to `submission.csv` at the root of the repository.

Computing the RoBERTa embeddings for ~260k texts is the expensive step: a GPU is strongly recommended. Embeddings are cached in `data/*.npy`, so later runs skip this step.

## Authors

- Lucas Zaidat
- Antoine Thuillier
- Ilyes Degardin

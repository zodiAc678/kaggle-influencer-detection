# Exploration

Approaches tried before the [final pipeline](../final_solution.ipynb), roughly in chronological order. They are kept to document the process; outputs were cleared and some notebooks may need minor adjustments to run again.

| Notebook | Approach |
|---|---|
| `01_baseline_tfidf_logreg.ipynb` | Dummy classifier and TF-IDF + Logistic Regression baselines (~0.63 validation accuracy) |
| `02_bertweet_finetuning.ipynb` | Fine-tuning [`vinai/bertweet-base`](https://huggingface.co/vinai/bertweet-base) on tweet text only |
| `03_bertweet_finetuning_v2.ipynb` | BERTweet with a custom head fusing text and scaled numerical features |
| `04_bertweet_finetuning_v3.ipynb` | Further BERTweet + features iteration |
| `05_bertweet_with_features/` | BERTweet fine-tuning where user features are injected **as text** into the input (e.g. `FOLLOWERS_COUNT=1234`, or categorical tokens like `major_influencer`) |
| `06_mlp_features_tfidf.ipynb` | MLP on handcrafted features + TF-IDF/SVD |
| `07_xgboost_features_tfidf.ipynb` | XGBoost on handcrafted features + TF-IDF/SVD |
| `08_xgboost_features_tfidf_v2.ipynb` | Variant of the above |
| `09_oof_stacking_transformers_gbdt.ipynb` | 5-fold out-of-fold stacking: RoBERTa, DistilBERT, XGBoost, LightGBM, CatBoost + neural meta-learner |

Main takeaway: text-only transformers plateaued far below models using user metadata; the best results came from combining engineered features with frozen transformer embeddings in gradient-boosted trees, which led to the final solution.

## About `05_bertweet_with_features/`

- `extract_features.py` defines several ways of turning user metadata into text (`extract_user_features`, `extract_full_features_new`, …).
- `main.py` selects the feature-extraction function and the number of training steps, then executes the notebook (which imports these settings from `main`). Run it from inside this folder: `python main.py`.

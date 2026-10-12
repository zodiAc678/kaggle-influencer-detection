# Data

The dataset is **not included** in this repository (size: ~1.1 GB, and redistribution is governed by the competition rules).

Download it from the Kaggle competition page and place the files here:

```
data/
├── train.jsonl         # 154,914 labelled tweets
└── kaggle_test.jsonl   # 103,380 unlabelled tweets
```

## Format

Each line is one tweet as a JSON object, following the Twitter API v1.1 tweet format.

| Field | Type | Description |
|---|---|---|
| `challenge_id` | int | Tweet identifier used for the submission |
| `label` | int | **Target** (train only): `1` = Influencer, `0` = Observer |
| `text` | str | Tweet text (may be truncated) |
| `extended_tweet` | dict | Contains `full_text` when the tweet is truncated |
| `user` | dict | Author profile: `followers_count`, `friends_count`, `listed_count`, `statuses_count`, `favourites_count`, `verified`, `description`, `created_at`, `location`, `url`, `profile_banner_url`, … |
| `source` | str | Client used to post (HTML link: iPhone, Android, Web, TweetDeck…) |
| `entities` / `extended_entities` | dict | Hashtags, mentions, URLs, media |
| `retweet_count`, `favorite_count`, `reply_count`, `quote_count` | int | Engagement counters |
| `in_reply_to_status_id`, `is_quote_status`, `quoted_status` | — | Reply / quote information |
| `created_at`, `timestamp_ms`, `lang`, `place`, … | — | Other tweet metadata (33 columns in total in train, 32 in test) |

## Generated files

Running the final notebook caches the RoBERTa embeddings here (`train_embeddings.npy`, `test_embeddings.npy`). They are ignored by git.

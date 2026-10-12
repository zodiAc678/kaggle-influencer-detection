from datetime import datetime
import pandas as pd


# ===========================================================
# Fonction de base
def extract_user_features(row):

    USER_FEATURES = [
    "followers_count",
    "friends_count",
    "listed_count",
    "statuses_count",
    "favourites_count"]

    user = row.get("user", {})

    # Certains dumps n'ont pas tous les champs → défaut = -1
    features = {f: user.get(f, -1) for f in USER_FEATURES}

    # Construction d'un bloc texte injecté dans le tweet
    feature_text = " ".join([f"{k.upper()}={v}" for k, v in features.items()])
    return feature_text

# ===========================================================

def categorize_followers(n):
    if n < 0:
        return "unknown_followers"
    if n < 50:
        return "new_user"
    if n < 300:
        return "small_account"
    if n < 5000:
        return "medium_account"
    if n < 50000:
        return "local_influencer"
    return "major_influencer"

def categorize_ratio(r):
    if r < 0:
        return "unknown_ratio"
    if r < 0.5:
        return "very_low_ratio"
    if r < 1:
        return "low_ratio"
    if r < 5:
        return "medium_ratio"
    if r < 20:
        return "strong_ratio"
    return "extremely_strong_ratio"

def categorize_source(source_string):
    if not isinstance(source_string, str):
        return "unknown_source"

    s = source_string.lower()
    if "iphone" in s or "android" in s:
        return "mobile_client"
    if "web" in s or "browser" in s:
        return "web_client"
    if "tweetdeck" in s or "hootsuite" in s:
        return "pro_client"
    if "api" in s:
        return "api_client"

    return "other_client"

def categorize_activity(n):
    if n < 0:
        return "unknown_activity"
    if n < 200:
        return "low_activity"
    if n < 2000:
        return "medium_activity"
    if n < 10000:
        return "high_activity"
    return "very_high_activity"

def categorize_age(years):
    if years < 0:
        return "unknown_age"
    if years < 1:
        return "very_recent_account"
    if years < 3:
        return "recent_account"
    if years < 7:
        return "established_account"
    return "old_account"

def extract_full_features_new(row):
    user = row.get("user", {})

    followers = user.get("followers_count", -1)
    friends   = user.get("friends_count", -1)
    listed    = user.get("listed_count", -1)
    statuses  = user.get("statuses_count", -1)
    favs      = user.get("favourites_count", -1)
    verified  = user.get("verified", False)
    source    = row.get("source", "")

    # Ratios
    ratio_ff = followers / (friends + 1) if friends > 0 else -1
    ratio_fs = followers / (statuses + 1) if statuses > 0 else -1
    ratio_lf = listed / (followers + 1) if followers > 0 else -1

    # Catégorisation simple (les meilleures pour la perf)
    foll_cat  = categorize_followers(followers)
    ratio_cat = categorize_ratio(ratio_ff)
    act_cat   = categorize_activity(statuses)
    verif_str = "verified" if verified else "not_verified"

    # Features du tweet
    entities = row.get("entities", {})
    n_hashtags = len(entities.get("hashtags", []))
    n_urls     = len(entities.get("urls", []))
    n_mentions = len(entities.get("user_mentions", []))

    # Médias ?
    ext = row.get("extended_entities", None)
    if isinstance(ext, dict):
        media_list = ext.get("media", [])
        has_media = 1 if isinstance(media_list, list) and len(media_list) > 0 else 0
    else:
        has_media = 0

    # Quote / reply
    is_quote = int(row.get("is_quote_status", False))
    is_reply = int(bool(row.get("in_reply_to_user_id")))

    # ==========================
    # Construction du bloc texte compact
    # ==========================
    profile_text = (
        f"[U verif={verif_str} foll={foll_cat} act={act_cat} "
        f"ratio_ff={ratio_ff:.2f} ratio_fs={ratio_fs:.2f} ratio_lf={ratio_lf:.2f} "
        f"hashtags={n_hashtags} urls={n_urls} mentions={n_mentions} "
        f"media={has_media} quote={is_quote} reply={is_reply}]"
    )

    return profile_text


def extract_full_features(row):
    user = row.get("user", {})

    followers = user.get("followers_count", -1)
    friends   = user.get("friends_count", -1)
    listed    = user.get("listed_count", -1)
    statuses  = user.get("statuses_count", -1)
    favs      = user.get("favourites_count", -1)
    verified  = user.get("verified", False)
    source    = row.get("source", "")

    # Ratios
    ratio_ff = followers / (friends + 1) if friends > 0 else -1
    ratio_fs = followers / (statuses + 1) if statuses > 0 else -1
    ratio_lf = listed / (followers + 1) if followers > 0 else -1

    # Age du compte
    created_at = user.get("created_at", None)
    if created_at:
        try:
            created = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
            age_years = (datetime.now(created.tzinfo) - created).days / 365
        except:
            age_years = -1
    else:
        age_years = -1

    # Catégorisation
    foll_cat  = categorize_followers(followers)
    ratio_cat = categorize_ratio(ratio_ff)
    act_cat   = categorize_activity(statuses)
    age_cat   = categorize_age(age_years)
    src_cat   = categorize_source(source)
    verif_str = "verified" if verified else "not_verified"

    # Features du tweet
    entities = row.get("entities", {})
    n_hashtags = len(entities.get("hashtags", []))
    n_urls     = len(entities.get("urls", []))
    n_mentions = len(entities.get("user_mentions", []))

    # Médias ?
    has_media = 0
    ext = row.get("extended_entities", None)
    # Certain rows have NaN → convert to empty dict
    if isinstance(ext, dict):
        media_list = ext.get("media", [])
        has_media = 1 if isinstance(media_list, list) and len(media_list) > 0 else 0
    else:
        has_media = 0
        
    # Quote / reply
    is_quote = row.get("is_quote_status", False)
    is_reply = bool(row.get("in_reply_to_user_id"))

    # Geo
    geo_enabled = user.get("geo_enabled", False)
    place_exists = 1 if row.get("place") else 0

    # ================
    # Features du QUOTED USER (si présent)
    # ================
    quoted = row.get("quoted_status")
    if isinstance(quoted, dict) and "user" in quoted:
        q_user = quoted["user"]
        q_followers = q_user.get("followers_count", -1)
        q_verified  = q_user.get("verified", False)

        quoted_user_desc = (
            f"Quoted user has {q_followers} followers "
            f"and is {'verified' if q_verified else 'not_verified'}."
        )
    else:
        quoted_user_desc = "No quoted user."

    # ==========================
    # Construction du bloc texte optimisé
    # ==========================
    profile_text = (
        f"[U verif={int(verified)} "
        f"foll={followers} fr={friends} list={listed} stat={statuses} fav={favs} "
        f"ff={ratio_ff:.2f} fs={ratio_fs:.2f} lf={ratio_lf:.2f} "
        f"hashtags={n_hashtags} urls={n_urls} mentions={n_mentions} "
        f"media={has_media} quote={int(is_quote)} reply={int(is_reply)}]"
    )

    return profile_text


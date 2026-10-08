from urllib.parse import urlparse
import re

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rb.gy", "rebrand.ly", "shorturl.at",
    "tiny.cc", "lnkd.in"
}
KEYWORDS = {
    "login", "verify", "verification", "secure", "update", "account",
    "password", "confirm", "bank", "wallet", "signin", "claim",
    "bonus", "payment", "invoice", "unlock", "support"
}
FEATURE_NAMES = [
    "url_length", "dot_count", "has_at", "https", "ip_address",
    "suspicious_keywords", "shortener", "hyphens", "subdomain_depth",
    "double_slash", "query_length", "digit_count"
]

def normalize_url(raw: str) -> str:
    raw = str(raw or "").strip()
    if not raw:
        raise ValueError("Please enter a URL.")
    if not re.match(r"^https?://", raw, flags=re.I):
        raw = "http://" + raw
    return raw

def extract_url_features(raw: str):
    url = normalize_url(raw)
    p = urlparse(url)
    host = (p.hostname or "").lower()
    labels = [x for x in host.split(".") if x]
    is_ip = bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host))
    lower = url.lower()
    keyword_count = sum(1 for k in KEYWORDS if k in lower)
    shortener = host in SHORTENERS or any(host.endswith("." + d) for d in SHORTENERS)
    remainder = url.split("://", 1)[1] if "://" in url else url
    path_query = remainder.split("/", 1)[1] if "/" in remainder else ""
    values = [
        len(url),
        max(len(labels) - 1, 0),
        int("@" in url),
        int(p.scheme.lower() == "https"),
        int(is_ip),
        keyword_count,
        int(shortener),
        host.count("-"),
        max(len(labels) - 2, 0),
        int("//" in path_query),
        len(p.query),
        sum(c.isdigit() for c in url),
    ]
    return {
        "url": url, "host": host or "—", "length": len(url),
        "dots": max(len(labels)-1,0), "has_at": "@" in url,
        "https": p.scheme.lower() == "https", "ip": is_ip,
        "keyword_count": keyword_count, "shortener": shortener,
        "hyphens": host.count("-"), "subdomain_depth": max(len(labels)-2,0),
        "double_slash": "//" in path_query, "query_length": len(p.query),
        "digit_count": sum(c.isdigit() for c in url), "values": values,
        "feature_names": FEATURE_NAMES,
    }

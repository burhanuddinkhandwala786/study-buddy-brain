import os
import re
import json
import hmac
import base64
import time
import sqlite3
import hashlib
import secrets
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Optional

from PIL import Image
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import httpx
from google import genai
from google.genai import types

# Decompression-bomb and payload caps
Image.MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGE_B64_CHARS = 5_500_000  # ~4 MB decoded image cap
MAX_TEXT_CHARS = 25_000

app = FastAPI(title="Cursor Companion Brain", version="7.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://burhanuddinkhandwala786.github.io/cursor-companion/"],
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["Content-Type", "X-Device-ID", "X-Admin-Secret"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


# Fast HTTP clients prevent 20-30s provider stalls
_HTTP_CLIENT = httpx.Client(timeout=8.5)
_REDIS_CLIENT = httpx.Client(timeout=1.5)
_HEDGE_POOL = ThreadPoolExecutor(max_workers=24)

IMAGE_CACHE = {}
CACHE_TTL = 1800

QUIZ_CACHE = {}
QUIZ_CACHE_TTL = 7200

KEY_COOLDOWN = {}
_KEY_CYCLE = itertools.count()

_LOCAL_BURST = {}
_LOCAL_BURST_LOCK = threading.Lock()

_GEMINI_CLIENTS = {}
_GEMINI_CLIENTS_LOCK = threading.Lock()
_DB_LOCK = threading.Lock()
_DB_PATH = "/tmp/companion_brain_cache_v13.db"

_DISCOVERED_MODELS = {}
_DISCOVERY_TTL = 21600  # 6 hours

_COMMUNITY_KEYS = {
    "groq": [],
    "gemini": [],
    "nvidia": [],
    "openrouter": [],
}
_PRO_DEVICES = set()
_COMMUNITY_LOCK = threading.Lock()

# --- Verified Active Zero-Card Models (Nov 2026) ---
GROQ_TEXT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]
GROQ_VISION_MODELS = [
    "qwen/qwen3.8-27b",
]

GEMINI_FAST_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]
GEMINI_DEEP_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]

CLOUDFLARE_TEXT_MODELS = [
    "@cf/google/gemma-4-26b-a4b-it",
    "@cf/qwen/qwen3.8-27b",
    "@cf/openai/gpt-oss-120b",
    "@cf/zai-org/glm-4.7-flash",
]
CLOUDFLARE_VISION_MODELS = [
    "@cf/google/gemma-4-26b-a4b-it",
    "@cf/qwen/qwen3.8-27b",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
]

NVIDIA_TEXT_MODELS = [
    "nvidia/nemotron-3.5-bolt-e4b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "mistralai/mistral-large-4-1t-a52b",
]
NVIDIA_VISION_MODELS = [
    "nvidia/nemotron-3.5-bolt-e4b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "mistralai/mistral-large-4-1t-a52b",
]

OPENROUTER_TEXT_MODELS = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
]
OPENROUTER_VISION_MODELS = [
    "openrouter/free",
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-31b-it:free",
]

# Optional BYOK-only providers (never required in shared free pool)
CEREBRAS_TEXT_MODELS = ["gpt-oss-120b", "qwen-3.8-27b"]
CEREBRAS_VISION_MODELS = ["qwen-3.8-27b"]
MISTRAL_TEXT_MODELS = ["mistral-small-latest"]
MISTRAL_VISION_MODELS = ["mistral-small-latest"]

PROVIDER_META = {
    "groq": {
        "env": "GROQ_API_KEY",
        "chat_url": "https://api.groq.com/openai/v1/chat/completions",
        "models_url": "https://api.groq.com/openai/v1/models",
        "text": GROQ_TEXT_MODELS,
        "vision": GROQ_VISION_MODELS,
    },
    "nvidia": {
        "env": "NVIDIA_API_KEY",
        "chat_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "models_url": "https://integrate.api.nvidia.com/v1/models",
        "text": NVIDIA_TEXT_MODELS,
        "vision": NVIDIA_VISION_MODELS,
    },
    "openrouter": {
        "env": "OPENROUTER_API_KEY",
        "chat_url": "https://openrouter.ai/api/v1/chat/completions",
        "models_url": "https://openrouter.ai/api/v1/models",
        "text": OPENROUTER_TEXT_MODELS,
        "vision": OPENROUTER_VISION_MODELS,
    },
    "cerebras": {
        "env": "CEREBRAS_API_KEY",
        "chat_url": "https://api.cerebras.ai/v1/chat/completions",
        "models_url": "https://api.cerebras.ai/v1/models",
        "text": CEREBRAS_TEXT_MODELS,
        "vision": CEREBRAS_VISION_MODELS,
    },
    "mistral": {
        "env": "MISTRAL_API_KEY",
        "chat_url": "https://api.mistral.ai/v1/chat/completions",
        "models_url": "https://api.mistral.ai/v1/models",
        "text": MISTRAL_TEXT_MODELS,
        "vision": MISTRAL_VISION_MODELS,
    },
}

PROMPT = """You are an elite academic, competitive-exam, and engineering verification engine.
Accuracy is #1. You MUST fill the "verification" field FIRST before choosing "cursor_pill":
1. Focus ONLY on the primary/centered question.
2. Read the ENTIRE question carefully (watch for NOT, INCORRECT, EXCEPT, and unit conversions).
3. Independently verify the factual or mathematical answer BEFORE looking at any highlighted/selected radio buttons.
4. COMPOUND OPTIONS CHECK: Read ALL options (A/B/C/D/E) before answering. If two individual options are true and a combined option exists ("both A and B", "both B and C", "all of the above", "none of the above"), you MUST select the combined option.
5. Only if the question stem or options are physically cut off or illegible, return cursor_pill "Unclear crop — retry".

Return ONLY valid JSON:
{"verification":"concise step-by-step check","detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"","pointer_xy":null}

Strict Output Rules:
- IF ANY OPTIONS (A/B/C/D/E or 1/2/3/4) ARE PRESENT: detected_type="MCQ", cursor_pill = exact option letter + text (e.g. "B. steel" or "D. both b and c"), copy_value="", expanded_note="".
- DIRECT_VALUE / CODE_FIX (no options present): cursor_pill = concise verified answer or fixed code line; copy_value = raw answer; expanded_note="".
- CONCEPT_EXPLAIN (only when user explicitly asks "explain" or "why" in follow-up): cursor_pill = 1-line summary; expanded_note = 2 short bullet points."""

V3_SOLVE_PROMPT = """You are an elite academic, competitive-exam, and engineering verification engine.
Accuracy is #1. You MUST fill the "verification" field FIRST before choosing "cursor_pill":
1. Focus ONLY on the primary/centered question.
2. Read the ENTIRE question carefully (watch for NOT, INCORRECT, EXCEPT, and unit conversions).
3. Independently verify the factual or mathematical answer BEFORE looking at any highlighted/selected radio buttons.
4. COMPOUND OPTIONS CHECK: Read ALL options (A/B/C/D/E) before answering. If two individual options are true and a combined option exists ("both A and B", "both B and C", "all of the above", "none of the above"), you MUST select the combined option.
5. Only if the question stem or options are physically cut off or illegible, return cursor_pill "Unclear crop — retry" and confidence "low".

Return ONLY valid JSON:
{"verification":"concise step-by-step check","detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"...","confidence":"high"|"medium","pointer_xy":null}

Strict Output Rules:
- IF ANY OPTIONS (A/B/C/D/E or 1/2/3/4) ARE PRESENT: detected_type="MCQ", cursor_pill = exact option letter + text (e.g. "B. steel"), copy_value="", expanded_note = 1 short sentence explaining why.
- DIRECT_VALUE / CODE_FIX: cursor_pill = exact concise answer; copy_value = raw answer; expanded_note = 1 short sentence.
- CONCEPT_EXPLAIN: cursor_pill = 1-line takeaway; copy_value = same takeaway; expanded_note = 3 short bullet points."""

V3_REWRITE_PROMPT = """Rewrite the provided text clearly and politely.
Output ONLY a single raw JSON object:
{"verification":"tone checked","detected_type":"REWRITE","cursor_pill":"...","copy_value":"...","expanded_note":"...","confidence":"high","pointer_xy":null}
Rules: cursor_pill = 1-line preview (max 65 chars); copy_value = complete rewritten text; expanded_note = brief tone note."""

V3_EXPLAIN_PROMPT = """Explain the provided concept, code, or error clearly in plain English.
Output ONLY a single raw JSON object:
{"verification":"facts checked","detected_type":"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"...","confidence":"high","pointer_xy":null}
Rules: cursor_pill = 1-line takeaway (max 70 chars); copy_value = same takeaway; expanded_note = 3 short bullet points."""


# --- Authenticated Encryption at Rest (Stdlib Encrypt-then-MAC, Zero Extra Deps) ---
def _get_master_key() -> bytes:
    seed = (
        os.environ.get("APP_ENCRYPTION_SECRET", "").strip()
        or os.environ.get("ADMIN_SECRET", "").strip()
        or os.environ.get("UPSTASH_REDIS_REST_TOKEN", "").strip()
        or "cursor-companion-default-v7-key-material"
    )
    return hashlib.sha256(seed.encode("utf-8")).digest()


def _keystream(enc_key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(hashlib.sha256(enc_key + nonce + counter.to_bytes(4, "big")).digest())
        counter += 1
    return bytes(out[:length])


def _encrypt_secret(plaintext: str) -> str:
    master = _get_master_key()
    enc_k = hashlib.sha256(master + b":enc").digest()
    mac_k = hashlib.sha256(master + b":mac").digest()
    nonce = secrets.token_bytes(16)
    pt = plaintext.encode("utf-8")
    ks = _keystream(enc_k, nonce, len(pt))
    ct = bytes(a ^ b for a, b in zip(pt, ks))
    tag = hmac.new(mac_k, nonce + ct, hashlib.sha256).digest()
    return "enc:v1:" + base64.urlsafe_b64encode(nonce + tag + ct).decode("ascii")


def _decrypt_secret(blob: str) -> str:
    if not isinstance(blob, str) or not blob:
        return ""
    if not blob.startswith("enc:v1:"):
        return blob
    try:
        raw = base64.urlsafe_b64decode(blob[7:].encode("ascii"))
        if len(raw) < 49:
            return ""
        nonce, tag, ct = raw[:16], raw[16:48], raw[48:]
        master = _get_master_key()
        enc_k = hashlib.sha256(master + b":enc").digest()
        mac_k = hashlib.sha256(master + b":mac").digest()
        expected = hmac.new(mac_k, nonce + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            return ""
        ks = _keystream(enc_k, nonce, len(ct))
        return bytes(a ^ b for a, b in zip(ct, ks)).decode("utf-8")
    except Exception:
        return ""


def _require_admin(request: Request, secret: Optional[str] = None):
    expected = os.environ.get("ADMIN_SECRET", "").strip()
    if not expected:
        return
    provided = (request.headers.get("X-Admin-Secret") or secret or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=403, detail="Admin secret required")


def _get_client_ip_hash(request: Optional[Request]) -> str:
    if not request:
        return ""
    xff = request.headers.get("X-Forwarded-For", "")
    ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else "")
    if not ip:
        return ""
    return hashlib.sha256(_get_master_key() + ip.encode("utf-8")).hexdigest()[:20]


# --- Persistent Upstash Redis REST Store (Single-RTT Pipeline) ---
def _redis_cfg():
    url = os.environ.get("UPSTASH_REDIS_REST_URL", "").strip().rstrip("/")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "").strip()
    return (url, token) if (url and token) else (None, None)


def _redis_cmd(*args):
    url, token = _redis_cfg()
    if not url:
        return None
    try:
        res = _REDIS_CLIENT.post(
            url,
            json=list(args),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        if res.status_code == 200:
            return res.json().get("result")
    except Exception:
        pass
    return None


def _redis_pipeline(commands: list):
    url, token = _redis_cfg()
    if not url or not commands:
        return None
    try:
        res = _REDIS_CLIENT.post(
            f"{url}/pipeline",
            json=commands,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        if res.status_code == 200:
            body = res.json()
            if isinstance(body, list):
                return [item.get("result") if isinstance(item, dict) else None for item in body]
    except Exception:
        pass
    return None


def _init_db():
    try:
        with _DB_LOCK:
            conn = sqlite3.connect(_DB_PATH)
            conn.execute("CREATE TABLE IF NOT EXISTS qcache (qhash TEXT PRIMARY KEY, ts REAL, payload TEXT)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS community_keys_v2 "
                "(key_hash TEXT PRIMARY KEY, provider TEXT, api_key_enc TEXT, device_id TEXT, ts REAL)"
            )
            conn.execute("CREATE TABLE IF NOT EXISTS pro_devices (device_id TEXT PRIMARY KEY, ts REAL)")
            rows = conn.execute("SELECT provider, api_key_enc FROM community_keys_v2").fetchall()
            pro_rows = conn.execute("SELECT device_id FROM pro_devices").fetchall()
            conn.commit()
            conn.close()
        with _COMMUNITY_LOCK:
            for prov, enc_key in rows:
                plain_k = _decrypt_secret(enc_key)
                if plain_k and prov in _COMMUNITY_KEYS and plain_k not in _COMMUNITY_KEYS[prov]:
                    _COMMUNITY_KEYS[prov].append(plain_k)
            for (dev,) in pro_rows:
                _PRO_DEVICES.add(dev)
    except Exception:
        pass

    for redis_key in ("cc:v2:community_pool", "cc:v1:community_pool"):
        raw_pool = _redis_cmd("GET", redis_key)
        if isinstance(raw_pool, str) and raw_pool:
            try:
                saved = json.loads(raw_pool)
                with _COMMUNITY_LOCK:
                    for prov, key_list in saved.get("keys", {}).items():
                        if prov in _COMMUNITY_KEYS and isinstance(key_list, list):
                            for blob in key_list:
                                plain_k = _decrypt_secret(blob)
                                if plain_k and plain_k not in _COMMUNITY_KEYS[prov]:
                                    _COMMUNITY_KEYS[prov].append(plain_k)
                    for dev in saved.get("pro_devices", []):
                        _PRO_DEVICES.add(dev)
            except Exception:
                pass


_init_db()


def _db_get(qhash: str, max_age: int = 86400) -> Optional[dict]:
    if not qhash:
        return None
    raw_r = _redis_cmd("GET", f"cc:q:{qhash}")
    if isinstance(raw_r, str) and raw_r:
        try:
            return json.loads(raw_r)
        except Exception:
            pass
    try:
        with _DB_LOCK:
            conn = sqlite3.connect(_DB_PATH)
            row = conn.execute("SELECT ts, payload FROM qcache WHERE qhash = ?", (qhash,)).fetchone()
            conn.close()
        if row and (time.time() - row[0]) < max_age:
            return json.loads(row[1])
    except Exception:
        pass
    return None


def _db_put(qhash: str, payload: dict):
    if not qhash:
        return
    serialized = json.dumps(payload)
    _redis_cmd("SET", f"cc:q:{qhash}", serialized, "EX", 86400)
    try:
        with _DB_LOCK:
            conn = sqlite3.connect(_DB_PATH)
            conn.execute(
                "INSERT OR REPLACE INTO qcache (qhash, ts, payload) VALUES (?, ?, ?)",
                (qhash, time.time(), serialized),
            )
            conn.execute("DELETE FROM qcache WHERE ts < ?", (time.time() - 86400,))
            conn.commit()
            conn.close()
    except Exception:
        pass


def _persist_community_pool(provider: str, api_key: str, device_id: str):
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    enc_val = _encrypt_secret(api_key)
    with _COMMUNITY_LOCK:
        if provider in _COMMUNITY_KEYS and api_key not in _COMMUNITY_KEYS[provider]:
            _COMMUNITY_KEYS[provider].append(api_key)
        if device_id:
            _PRO_DEVICES.add(device_id)
        encrypted_snapshot = {
            "keys": {k: [_encrypt_secret(val) for val in v] for k, v in _COMMUNITY_KEYS.items()},
            "pro_devices": list(_PRO_DEVICES),
        }
    try:
        with _DB_LOCK:
            conn = sqlite3.connect(_DB_PATH)
            conn.execute(
                "INSERT OR REPLACE INTO community_keys_v2 (key_hash, provider, api_key_enc, device_id, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (key_hash, provider, enc_val, device_id or "", time.time()),
            )
            if device_id:
                conn.execute(
                    "INSERT OR REPLACE INTO pro_devices (device_id, ts) VALUES (?, ?)",
                    (device_id, time.time()),
                )
            conn.commit()
            conn.close()
    except Exception:
        pass
    _redis_cmd("SET", "cc:v2:community_pool", json.dumps(encrypted_snapshot))


def _check_all_limits_pipelined(device_id: Optional[str], ip_hash: str, is_pro: bool) -> str:
    """Checks burst (10/min), daily device cap (10 Free / 150 Pro), and daily IP cap in 1 Redis RTT."""
    ident = device_id or ip_hash
    if not ident:
        return "ok"

    try:
        max_rpm = int(os.environ.get("BURST_RPM_LIMIT", "10"))
        dev_limit = int(os.environ.get("DAILY_PRO_LIMIT", "150") if is_pro else os.environ.get("DAILY_FREE_LIMIT", "10"))
        ip_limit = int(os.environ.get("DAILY_IP_LIMIT", "120"))
    except ValueError:
        max_rpm, dev_limit, ip_limit = 10, (150 if is_pro else 10), 120

    if _redis_cfg()[0]:
        minute_bucket = int(time.time() // 60)
        day = time.strftime("%Y%m%d", time.gmtime())
        cmds = []
        slots = {}

        burst_key = f"cc:burst:{ident}:{minute_bucket}"
        slots["burst"] = len(cmds)
        cmds.append(["INCR", burst_key])
        cmds.append(["EXPIRE", burst_key, 70, "NX"])

        if device_id:
            dev_key = f"cc:rl:{device_id}:{day}"
            slots["dev"] = len(cmds)
            cmds.append(["INCR", dev_key])
            cmds.append(["EXPIRE", dev_key, 86400, "NX"])

        if ip_hash:
            ip_key = f"cc:iprl:{ip_hash}:{day}"
            slots["ip"] = len(cmds)
            cmds.append(["INCR", ip_key])
            cmds.append(["EXPIRE", ip_key, 86400, "NX"])

        results = _redis_pipeline(cmds)
        if results:
            b_cnt = results[slots["burst"]]
            if isinstance(b_cnt, int) and b_cnt > max_rpm:
                return "burst"
            if "dev" in slots:
                d_cnt = results[slots["dev"]]
                if isinstance(d_cnt, int) and d_cnt > dev_limit:
                    return "quota"
            if "ip" in slots:
                i_cnt = results[slots["ip"]]
                if isinstance(i_cnt, int) and i_cnt > ip_limit:
                    return "quota"
        return "ok"

    # Fallback local memory burst protection if Redis is not configured
    now = time.time()
    with _LOCAL_BURST_LOCK:
        timestamps = [t for t in _LOCAL_BURST.get(ident, []) if now - t < 60]
        if len(timestamps) >= max_rpm:
            _LOCAL_BURST[ident] = timestamps
            return "burst"
        timestamps.append(now)
        _LOCAL_BURST[ident] = timestamps
        if len(_LOCAL_BURST) > 1000:
            _LOCAL_BURST.pop(next(iter(_LOCAL_BURST)))
    return "ok"


def get_env_keys(env_name: str):
    return [
        key.strip().strip('"').strip("'")
        for key in os.environ.get(env_name, "").split(",")
        if key.strip().strip('"').strip("'")
    ]


def get_pool_keys(provider: str):
    env_name = "GEMINI_API_KEY" if provider == "gemini" else PROVIDER_META[provider]["env"]
    keys = get_env_keys(env_name)
    with _COMMUNITY_LOCK:
        for ck in _COMMUNITY_KEYS.get(provider, []):
            if ck not in keys:
                keys.append(ck)
    return keys


def _get_cloudflare_cfg():
    raw_acc = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip().strip('"').strip("'")
    keys = get_env_keys("CLOUDFLARE_API_KEY")
    if not raw_acc and not keys:
        return ("", [], "not_configured")
    if not keys:
        return ("", [], "MISSING_API_KEY")
    if re.fullmatch(r"[0-9a-fA-F]{32}", raw_acc):
        return (raw_acc, keys, "ok")
    match = re.search(r"(?<![0-9a-fA-F])([0-9a-fA-F]{32})(?![0-9a-fA-F])", raw_acc)
    if match:
        return (match.group(1), keys, "ok")
    return ("", [], "INVALID_ACCOUNT_ID (must be raw 32-char hex Account ID from dash.cloudflare.com)")


def parse_user_keys(raw_user_keys: Optional[str]):
    cleaned_raw = re.sub(r"\b(?:bearer|api_key|key)\s*[:=]?\s*", "", raw_user_keys or "", flags=re.I)
    return [
        key.strip().strip('"').strip("'")
        for key in re.split(r"[\s,;]+", cleaned_raw)
        if key.strip().strip('"').strip("'")
    ]


def detect_key_provider(api_key: str) -> tuple:
    if api_key.lower().startswith("mistral:"):
        return ("mistral", api_key.split(":", 1)[1].strip())
    if api_key.startswith("csk-"):
        return ("cerebras", api_key)
    if api_key.startswith(("AIza", "AQ.")):
        return ("gemini", api_key)
    if api_key.startswith("gsk_"):
        return ("groq", api_key)
    if api_key.startswith("nvapi-"):
        return ("nvidia", api_key)
    if api_key.startswith("sk-or-"):
        return ("openrouter", api_key)
    if api_key.startswith("sk-"):
        return ("openai", api_key)
    return ("unknown", api_key)


def normalize_quiz_hash(text: str) -> str:
    s = re.sub(
        r"^\s*(?:q(?:uestion)?\s*\d+[\.\)\:\-]*|\d+[\.\)\:\-]+|\[\d+\s*marks?\])\s*",
        "",
        (text or "").strip(),
        flags=re.I,
    )
    words = re.findall(r"[a-z0-9]+", s.lower())
    cleaned = "".join(words)
    return hashlib.md5(f"v13:{cleaned}".encode()).hexdigest() if (len(cleaned) >= 30 and len(words) >= 5) else ""


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None
    user_api_key: Optional[str] = None


class V3AnalyzeRequest(AnalyzeRequest):
    action_mode: Optional[str] = "solve"
    device_id: Optional[str] = None
    custom_prompt: Optional[str] = None


class ContributeKeyRequest(BaseModel):
    api_key: str
    device_id: Optional[str] = None


def _get_client(api_key: str):
    client = _GEMINI_CLIENTS.get(api_key)
    if client is None:
        with _GEMINI_CLIENTS_LOCK:
            client = _GEMINI_CLIENTS.get(api_key)
            if client is None:
                # 20s deadline satisfies Google's >=10s rule and avoids false X-Server-Timeout 504s
                client = genai.Client(api_key=api_key, http_options={"timeout": 20000})
                _GEMINI_CLIENTS[api_key] = client
    return client


# --- Full 5-Provider Self-Healing Live Model Auto-Discovery ---
def _discover_live_models(provider_tag: str, api_key: str, want_vision: bool = False) -> list:
    meta = PROVIDER_META.get(provider_tag)
    if not meta:
        return []
    cache_id = f"{provider_tag}:{'img' if want_vision else 'txt'}"
    cached = _DISCOVERED_MODELS.get(cache_id)
    if cached and (time.time() - cached[0]) < _DISCOVERY_TTL:
        return cached[1]

    try:
        r = _HTTP_CLIENT.get(
            meta["models_url"],
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if r.status_code != 200:
            return []
        items = r.json().get("data", [])
        skip_words = (
            "whisper", "tts", "audio", "embed", "guard", "moderation",
            "ocr", "transcribe", "realtime", "image-gen", "rerank", "safeguard",
        )
        found = []
        for item in items:
            mid = str(item.get("id", "")).strip()
            low = mid.lower()
            if not mid or any(w in low for w in skip_words):
                continue
            if provider_tag == "openrouter" and not (low.endswith(":free") or low == "openrouter/free"):
                continue
            if want_vision:
                if any(v in low for v in ("vl", "vision", "qwen", "gemma-4", "mistral-large-4", "bolt", "free")):
                    found.append(mid)
            else:
                found.append(mid)
        found = found[:4]
        if found:
            _DISCOVERED_MODELS[cache_id] = (time.time(), found)
        return found
    except Exception:
        return []


def _discover_gemini_models(api_key: str) -> list:
    """Auto-fetches active Gemini Flash models if hardcoded slugs ever retire."""
    cache_id = "gemini:live"
    cached = _DISCOVERED_MODELS.get(cache_id)
    if cached and (time.time() - cached[0]) < _DISCOVERY_TTL:
        return cached[1]
    try:
        r = _HTTP_CLIENT.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=100"
        )
        if r.status_code != 200:
            return []
        found = []
        for m in r.json().get("models", []):
            name = str(m.get("name", "")).replace("models/", "").strip()
            methods = m.get("supportedGenerationMethods", [])
            low = name.lower()
            if (
                "generateContent" in methods
                and "gemini" in low
                and "flash" in low
                and not any(x in low for x in ("embed", "image", "audio", "tts", "live", "exp", "tuning", "2.0", "1.5"))
            ):
                found.append(name)
        found.sort(reverse=True)
        found = found[:4]
        if found:
            _DISCOVERED_MODELS[cache_id] = (time.time(), found)
        return found
    except Exception:
        return []


def _discover_cloudflare_models(account_id: str, api_key: str, want_vision: bool = False) -> list:
    """Auto-fetches active Cloudflare Workers AI models if a hardcoded slug retires (404/410)."""
    cache_id = f"cloudflare:{'img' if want_vision else 'txt'}"
    cached = _DISCOVERED_MODELS.get(cache_id)
    if cached and (time.time() - cached[0]) < _DISCOVERY_TTL:
        return cached[1]
    try:
        r = _HTTP_CLIENT.get(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search?task=Text%20Generation",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if r.status_code != 200:
            return []
        found = []
        for item in r.json().get("result", []):
            mid = str(item.get("name", "")).strip()
            low = mid.lower()
            if not mid.startswith("@cf/"):
                continue
            if any(bad in low for bad in ("lora", "guard", "embed", "tiny")):
                continue
            if want_vision:
                if any(v in low for v in ("gemma-4", "qwen3.8", "llama-4-scout", "vision", "vl")):
                    found.append(mid)
            else:
                if any(k in low for k in ("gemma-4", "qwen3.8", "gpt-oss", "glm-4", "llama-4", "llama-3.3")):
                    found.append(mid)
        found = found[:4]
        if found:
            _DISCOVERED_MODELS[cache_id] = (time.time(), found)
        return found
    except Exception:
        return []


def _get_live_or_default(cache_id: str, default_list: list) -> list:
    cached = _DISCOVERED_MODELS.get(cache_id)
    if cached and (time.time() - cached[0]) < _DISCOVERY_TTL and cached[1]:
        merged = list(cached[1])
        for m in default_list:
            if m not in merged:
                merged.append(m)
        return merged[:4]
    return list(default_list)


@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/healthz", methods=["GET", "HEAD"])
def health():
    now = time.time()
    cf_acc, cf_keys, cf_status = _get_cloudflare_cfg()
    with _COMMUNITY_LOCK:
        community_total = sum(len(v) for v in _COMMUNITY_KEYS.values())
        pro_count = len(_PRO_DEVICES)
    return {
        "status": "ok",
        "engine": "v7.2.0-selfheal",
        "groq_keys": len(get_pool_keys("groq")),
        "gemini_keys": len(get_pool_keys("gemini")),
        "cloudflare_keys": len(cf_keys) if cf_status == "ok" else 0,
        "cloudflare_status": cf_status,
        "nvidia_keys": len(get_pool_keys("nvidia")),
        "openrouter_keys": len(get_pool_keys("openrouter")),
        "community_keys": community_total,
        "pro_devices": pro_count,
        "persistent_cache": bool(_redis_cfg()[0]),
        "admin_protected": bool(os.environ.get("ADMIN_SECRET", "").strip()),
        "cached_items": len(IMAGE_CACHE) + len(QUIZ_CACHE),
        "active_cooldowns": sum(1 for exp in KEY_COOLDOWN.values() if exp > now),
    }


@app.get("/v3_1/config")
def remote_client_config():
    return {
        "latest_version": os.environ.get("APP_LATEST_VERSION", "3.1.0").strip(),
        "release_url": os.environ.get(
            "APP_RELEASE_URL",
            "https://github.com/burhanuddinkhandwala786/cursor-companion/releases/latest",
        ).strip(),
        "notice": os.environ.get("APP_NOTICE", "").strip(),
        "byok_models": {
            "groq_text": _get_live_or_default("groq:txt", GROQ_TEXT_MODELS),
            "groq_vision": _get_live_or_default("groq:img", GROQ_VISION_MODELS),
            "gemini": _get_live_or_default("gemini:live", GEMINI_FAST_MODELS),
            "nvidia_text": _get_live_or_default("nvidia:txt", NVIDIA_TEXT_MODELS),
            "nvidia_vision": _get_live_or_default("nvidia:img", NVIDIA_VISION_MODELS),
            "openrouter_text": _get_live_or_default("openrouter:txt", OPENROUTER_TEXT_MODELS),
            "openrouter_vision": _get_live_or_default("openrouter:img", OPENROUTER_VISION_MODELS),
            "cerebras_text": CEREBRAS_TEXT_MODELS,
            "cerebras_vision": CEREBRAS_VISION_MODELS,
            "mistral_text": MISTRAL_TEXT_MODELS,
            "mistral_vision": MISTRAL_VISION_MODELS,
        },
    }


@app.get("/diag")
def diagnose_keys(request: Request, secret: Optional[str] = Query(default=None)):
    _require_admin(request, secret)
    report = {
        "groq": "not_configured",
        "gemini": [],
        "cloudflare": "not_configured",
        "nvidia": "not_configured",
        "openrouter": "not_configured",
    }

    for tag in ("groq", "nvidia", "openrouter"):
        keys = get_pool_keys(tag)
        if not keys:
            continue
        meta = PROVIDER_META[tag]
        model = _get_live_or_default(f"{tag}:txt", meta["text"])[0]
        try:
            body = {"model": model, "messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 32}
            if tag == "groq" and "gpt-oss" in model:
                body["reasoning_effort"] = "low"
            r = _HTTP_CLIENT.post(
                meta["chat_url"],
                json=body,
                headers={"Authorization": f"Bearer {keys[0]}", "Content-Type": "application/json"},
            )
            report[tag] = "HEALTHY" if r.status_code == 200 else f"HTTP {r.status_code}: {r.text[:100]}"
        except Exception as exc:
            report[tag] = f"ERROR: {str(exc)[:100]}"

    cf_acc, cf_keys, cf_status = _get_cloudflare_cfg()
    if cf_status == "ok":
        cf_model = _get_live_or_default("cloudflare:txt", CLOUDFLARE_TEXT_MODELS)[0]
        try:
            r = _HTTP_CLIENT.post(
                f"https://api.cloudflare.com/client/v4/accounts/{cf_acc}/ai/v1/chat/completions",
                json={
                    "model": cf_model,
                    "messages": [{"role": "user", "content": "Reply OK"}],
                    "max_tokens": 24,
                },
                headers={"Authorization": f"Bearer {cf_keys[0]}", "Content-Type": "application/json"},
            )
            report["cloudflare"] = "HEALTHY" if r.status_code == 200 else f"HTTP {r.status_code}: {r.text[:100]}"
        except Exception as exc:
            report["cloudflare"] = f"ERROR: {str(exc)[:100]}"
    elif cf_status != "not_configured":
        report["cloudflare"] = f"CONFIG_ERROR: {cf_status}"

    gemini_diag_model = _get_live_or_default("gemini:live", GEMINI_FAST_MODELS)[0]
    for index, api_key in enumerate(get_pool_keys("gemini"), 1):
        try:
            client = _get_client(api_key)
            response = client.models.generate_content(
                model=gemini_diag_model,
                contents="Reply OK",
                config=types.GenerateContentConfig(max_output_tokens=20),
            )
            report["gemini"].append({
                "key": f"G{index} (...{api_key[-4:]})",
                "status": "HEALTHY",
                "reply": (response.text or "").strip(),
            })
        except Exception as exc:
            report["gemini"].append({
                "key": f"G{index} (...{api_key[-4:]})",
                "status": "ERROR",
                "reason": str(exc)[:120],
            })

    return report


@app.get("/selftest")
@app.get("/v3/selftest")
def self_test(request: Request, secret: Optional[str] = Query(default=None)):
    _require_admin(request, secret)
    results = {}
    t0 = time.time()
    probe = 'Return JSON: {"verification":"ok","detected_type":"DIRECT_VALUE","cursor_pill":"2","copy_value":"2","expanded_note":"","pointer_xy":null}'

    for tag in ("groq", "nvidia", "openrouter"):
        keys = get_pool_keys(tag)
        if not keys:
            results[f"{tag}_pipeline"] = {"status": "SKIPPED (not configured)"}
            continue
        meta = PROVIDER_META[tag]
        t_p = time.time()
        try:
            raw = call_openai_compat(
                endpoint=meta["chat_url"],
                api_key=keys[0],
                models=meta["text"],
                prompt_text=probe,
                provider_tag=tag,
                max_models=2,
            )
            results[f"{tag}_pipeline"] = {
                "status": "200 OK",
                "output": extract_json(raw).get("cursor_pill"),
                "ms": round((time.time() - t_p) * 1000),
            }
        except Exception as exc:
            results[f"{tag}_pipeline"] = {
                "status": "FAIL",
                "error": str(exc)[:120],
                "ms": round((time.time() - t_p) * 1000),
            }

    cf_acc, cf_keys, cf_status = _get_cloudflare_cfg()
    if cf_status == "ok":
        t_p = time.time()
        try:
            raw_cf = _call_cloudflare_pool(probe)
            results["cloudflare_pipeline"] = (
                {
                    "status": "200 OK",
                    "output": extract_json(raw_cf).get("cursor_pill"),
                    "ms": round((time.time() - t_p) * 1000),
                }
                if raw_cf
                else {"status": "FAIL", "error": "Cloudflare returned empty"}
            )
        except Exception as exc:
            results["cloudflare_pipeline"] = {"status": "FAIL", "error": str(exc)[:120]}
    else:
        results["cloudflare_pipeline"] = {"status": f"SKIPPED ({cf_status})"}

    gemini_keys = get_pool_keys("gemini")
    if gemini_keys:
        t_p = time.time()
        try:
            raw_gemini = call_gemini_keys(
                keys=gemini_keys,
                prompt_text='Return JSON: {"verification":"ok","detected_type":"DIRECT_VALUE","cursor_pill":"4","copy_value":"4","expanded_note":"","pointer_xy":null}',
                is_byok=False,
            )
            results["gemini_pipeline"] = {
                "status": "200 OK",
                "output": extract_json(raw_gemini).get("cursor_pill"),
                "ms": round((time.time() - t_p) * 1000),
            }
        except Exception as exc:
            results["gemini_pipeline"] = {
                "status": "FAIL",
                "error": str(exc)[:120],
                "ms": round((time.time() - t_p) * 1000),
            }

    byok_guard = _unified_execute(
        AnalyzeRequest(
            text_content="What is 2+2?",
            user_api_key="AIzaInvalidTestKey1, AIzaInvalidTestKey2",
        ),
        action_mode="solve",
        preserve_notes=False,
    )
    results["byok_isolation_guard"] = {
        "status": "200 OK" if byok_guard.get("key_source") == "BYOK" else "FAIL",
        "message": byok_guard.get("cursor_pill"),
    }

    mock_check = _unified_execute(
        V3AnalyzeRequest(text_content="TEST: UI alignment check"),
        action_mode="solve",
        preserve_notes=True,
    )
    results["zero_token_test_mode"] = {
        "status": "200 OK (0 tokens)" if mock_check.get("key_source") == "TEST" else "FAIL",
        "pill": mock_check.get("cursor_pill"),
    }
    results["elapsed_ms"] = round((time.time() - t0) * 1000)
    return results


@app.post("/v3_1/contribute-key")
def contribute_community_key(req: ContributeKeyRequest, request: Request):
    ip_hash = _get_client_ip_hash(request)
    if _check_all_limits_pipelined(req.device_id, ip_hash, is_pro=True) == "burst":
        return {"ok": False, "message": "Too many attempts — wait 1 minute"}

    keys = parse_user_keys(req.api_key)
    if not keys:
        return {"ok": False, "message": "Invalid key format"}

    prov, clean_key = detect_key_provider(keys[0])
    if prov not in _COMMUNITY_KEYS:
        return {"ok": False, "message": "Contribute a free Groq (gsk_), Gemini (AIza), NVIDIA (nvapi-), or OpenRouter (sk-or-) key"}

    try:
        if prov == "gemini":
            client = _get_client(clean_key)
            check_model = _get_live_or_default("gemini:live", GEMINI_FAST_MODELS)[0]
            client.models.generate_content(
                model=check_model,
                contents="Reply OK",
                config=types.GenerateContentConfig(max_output_tokens=16),
            )
        else:
            meta = PROVIDER_META[prov]
            check_model = _get_live_or_default(f"{prov}:txt", meta["text"])[0]
            body = {"model": check_model, "messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 24}
            if prov == "groq" and "gpt-oss" in check_model:
                body["reasoning_effort"] = "low"
            r = _HTTP_CLIENT.post(
                meta["chat_url"],
                json=body,
                headers={"Authorization": f"Bearer {clean_key}", "Content-Type": "application/json"},
            )
            if r.status_code not in (200, 429):
                return {"ok": False, "message": f"Key verification failed (HTTP {r.status_code})"}
    except Exception as exc:
        return {"ok": False, "message": f"Key check failed: {str(exc)[:80]}"}

    _persist_community_pool(prov, clean_key, req.device_id or "")
    return {
        "ok": True,
        "provider": prov,
        "tier": "PRO",
        "message": f"Pro Pool Unlocked ({prov.upper()} contributed ✓)",
    }


def extract_json(raw_text: str, preserve_notes: bool = False) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.S | re.I)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.S | re.I)
    cleaned = re.sub(r"```(?:json)?|```", "", cleaned, flags=re.I).strip()

    if not cleaned:
        raise ValueError("Empty model output")

    data = None

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if isinstance(parsed, dict) and "cursor_pill" in parsed:
            data = parsed
    except Exception:
        pass

    if data is None:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(cleaned):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(cleaned[idx:])
                    if isinstance(obj, dict) and "cursor_pill" in obj:
                        data = obj
                        break
                except Exception:
                    continue

    if data is None:
        pill_match = re.search(r'"cursor_pill"\s*:\s*"([^"]+)"', cleaned)
        if pill_match:
            pill_val = pill_match.group(1).strip()
            copy_match = re.search(r'"copy_value"\s*:\s*"([^"]*)"', cleaned)
            note_match = re.search(r'"expanded_note"\s*:\s*"([^"]*)"', cleaned)
            type_match = re.search(r'"detected_type"\s*:\s*"([^"]+)"', cleaned)
            conf_match = re.search(r'"confidence"\s*:\s*"([^"]+)"', cleaned)
            data = {
                "detected_type": type_match.group(1) if type_match else "DIRECT_VALUE",
                "cursor_pill": pill_val,
                "copy_value": copy_match.group(1) if copy_match else pill_val,
                "expanded_note": note_match.group(1) if note_match else "",
                "confidence": conf_match.group(1) if conf_match else "high",
                "pointer_xy": None,
            }

    if not isinstance(data, dict) or not str(data.get("cursor_pill", "")).strip():
        raise ValueError("Model returned prose/thinking instead of JSON")

    data.pop("verification", None)

    pill_str = str(data["cursor_pill"]).strip()
    data.setdefault("detected_type", "DIRECT_VALUE")
    data["cursor_pill"] = pill_str
    data.setdefault("copy_value", "")
    data.setdefault("expanded_note", "")
    data.setdefault("confidence", "high")
    data.setdefault("pointer_xy", None)

    if not preserve_notes and data.get("detected_type") in ("MCQ", "DIRECT_VALUE", "CODE_FIX"):
        data["expanded_note"] = ""

    return data


def _classify_gemini_error(error_text: str) -> str:
    lowered = error_text.lower()
    if (
        "503" in error_text
        or "504" in error_text
        or "unavailable" in lowered
        or "deadline_exceeded" in lowered
        or "timed out" in lowered
    ):
        return "overload"
    if "429" in error_text or "resource_exhausted" in lowered or "quota" in lowered:
        return "rate_limit"
    if (
        "401" in error_text
        or "402" in error_text
        or "403" in error_text
        or "permission_denied" in lowered
        or "api_key_invalid" in lowered
        or "api key not valid" in lowered
    ):
        return "forbidden"
    if "404" in error_text or "not_found" in lowered or "decommission" in lowered or "retired" in lowered:
        return "model_unavailable"
    return "other"


def _try_one_gemini(
    api_key: str,
    model_name: str,
    prompt_text: str,
    img_pil,
    use_thinking: bool,
    sys_prompt: str = PROMPT,
) -> str:
    cfg_kwargs = {
        "response_mime_type": "application/json",
        "max_output_tokens": 900,
    }
    if use_thinking:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")

    client = _get_client(api_key)
    contents = [sys_prompt, prompt_text] + ([img_pil] if img_pil is not None else [])

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        if response and response.text:
            extract_json(response.text, preserve_notes=True)
            return response.text
        raise RuntimeError("empty response")
    except Exception as exc:
        error_text = str(exc)
        if use_thinking and ("thinking" in error_text.lower() or "invalid_argument" in error_text.lower()):
            cfg_kwargs.pop("thinking_config", None)
            retry_response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            if retry_response and retry_response.text:
                extract_json(retry_response.text, preserve_notes=True)
                return retry_response.text
        raise


def call_gemini_keys(
    keys: list,
    prompt_text: str,
    img_pil: Optional[Image.Image] = None,
    deep: bool = False,
    is_byok: bool = False,
    sys_prompt: str = PROMPT,
) -> str:
    if not keys:
        raise RuntimeError("No Gemini keys configured")

    now = time.time()
    start_idx = next(_KEY_CYCLE) % len(keys)
    ordered_keys = keys[start_idx:] + keys[:start_idx]

    base_models = list(GEMINI_DEEP_MODELS if deep else GEMINI_FAST_MODELS)
    cached_live = _DISCOVERED_MODELS.get("gemini:live")
    if cached_live and (now - cached_live[0]) < _DISCOVERY_TTL:
        for m in cached_live[1]:
            if m not in base_models:
                base_models.append(m)

    last_error = "Gemini keys cooling down"
    attempts = 0
    max_attempts = 4 if is_byok else 2
    discovered_now = False
    idx = 0

    while idx < len(base_models):
        model_name = base_models[idx]
        idx += 1
        use_thinking = "lite" not in model_name
        for api_key in ordered_keys:
            if attempts >= max_attempts:
                break
            if KEY_COOLDOWN.get(api_key, 0) > now:
                continue
            model_cd_key = f"{api_key}:{model_name}"
            if KEY_COOLDOWN.get(model_cd_key, 0) > now:
                continue

            attempts += 1
            try:
                return _try_one_gemini(
                    api_key, model_name, prompt_text, img_pil, use_thinking, sys_prompt=sys_prompt
                )
            except Exception as exc:
                error_text = str(exc)
                last_error = f"{model_name}: {error_text[:100]}"
                err_type = _classify_gemini_error(error_text)

                if err_type == "overload":
                    KEY_COOLDOWN[model_cd_key] = time.time() + 60
                elif err_type == "rate_limit":
                    KEY_COOLDOWN[model_cd_key] = time.time() + 25
                elif err_type == "forbidden":
                    KEY_COOLDOWN[api_key] = time.time() + 3600
                elif err_type == "model_unavailable":
                    KEY_COOLDOWN[model_cd_key] = time.time() + 21600
                    if not discovered_now:
                        discovered_now = True
                        added = 0
                        for live_m in _discover_gemini_models(api_key):
                            if live_m not in base_models:
                                base_models.append(live_m)
                                added += 1
                        if added:
                            max_attempts += 1
        if attempts >= max_attempts:
            break

    if is_byok and all(KEY_COOLDOWN.get(k, 0) > time.time() for k in keys):
        raise RuntimeError("BYOK_INVALID: Invalid personal API key (Alt+K)")
    if is_byok:
        raise RuntimeError(f"BYOK_LIMIT: {last_error}")
    raise RuntimeError(last_error)


def call_openai_compat(
    endpoint: str,
    api_key: str,
    models: list,
    prompt_text: str,
    img_b64: Optional[str] = None,
    provider_tag: str = "compat",
    is_byok: bool = False,
    sys_prompt: str = PROMPT,
    max_models: int = 2,
) -> str:
    last_error = f"No {provider_tag} attempt succeeded"
    modality = "img" if img_b64 else "txt"
    tried = set()
    candidate_models = _get_live_or_default(f"{provider_tag}:{modality}", models)
    discovered_added = False
    attempts = 0
    attempt_cap = 6 if is_byok else max_models

    idx = 0
    while idx < len(candidate_models) and attempts < attempt_cap:
        model_id = candidate_models[idx]
        idx += 1
        if model_id in tried:
            continue
        tried.add(model_id)

        if KEY_COOLDOWN.get(f"{provider_tag}:{api_key}", 0) > time.time():
            break
        cooldown_key = f"{provider_tag}:{api_key}:{model_id}:{modality}"
        if KEY_COOLDOWN.get(cooldown_key, 0) > time.time():
            continue

        attempts += 1

        if img_b64:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{sys_prompt}\n\n{prompt_text}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }]
        else:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_text},
            ]

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if provider_tag == "openrouter":
            headers["HTTP-Referer"] = "https://study-buddy-brain.onrender.com"
            headers["X-Title"] = "Cursor Companion"

        payload_body = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 900,
        }
        if provider_tag == "groq" and "gpt-oss" in model_id:
            payload_body["reasoning_effort"] = "low"
        if provider_tag in ("groq", "cerebras", "mistral") and not img_b64:
            payload_body["response_format"] = {"type": "json_object"}

        try:
            response = _HTTP_CLIENT.post(endpoint, json=payload_body, headers=headers)
        except Exception as exc:
            last_error = f"{provider_tag} {model_id}: {str(exc)[:80]}"
            continue

        if response.status_code == 200:
            try:
                content = response.json()["choices"][0]["message"].get("content") or ""
                if content.strip():
                    extract_json(content, preserve_notes=True)
                    return content
            except Exception as parse_exc:
                last_error = f"{provider_tag} {model_id}: {str(parse_exc)[:60]}"
                continue

        if response.status_code == 400 and ("response_format" in payload_body or "reasoning_effort" in payload_body):
            payload_body.pop("response_format", None)
            payload_body.pop("reasoning_effort", None)
            try:
                retry_res = _HTTP_CLIENT.post(endpoint, json=payload_body, headers=headers)
                if retry_res.status_code == 200:
                    content = retry_res.json()["choices"][0]["message"].get("content") or ""
                    if content.strip():
                        extract_json(content, preserve_notes=True)
                        return content
            except Exception:
                pass

        last_error = f"{provider_tag} {model_id}: HTTP {response.status_code}"
        if response.status_code in (401, 402, 403):
            KEY_COOLDOWN[f"{provider_tag}:{api_key}"] = time.time() + (86400 if response.status_code == 402 else 3600)
            if is_byok:
                raise RuntimeError("BYOK_INVALID: Key invalid or credit exhausted (Alt+K)")
            break
        if response.status_code in (429, 503, 504):
            if provider_tag == "openrouter" and "per-day" in response.text.lower():
                KEY_COOLDOWN[f"{provider_tag}:{api_key}"] = time.time() + 21600
                break
            KEY_COOLDOWN[cooldown_key] = time.time() + (45 if response.status_code >= 500 else 25)
        elif response.status_code in (400, 404, 410):
            KEY_COOLDOWN[cooldown_key] = time.time() + 21600
            if not discovered_added and idx >= len(candidate_models):
                discovered_added = True
                if provider_tag == "cloudflare":
                    cf_acc, _, _ = _get_cloudflare_cfg()
                    live_found = _discover_cloudflare_models(cf_acc, api_key, want_vision=bool(img_b64)) if cf_acc else []
                else:
                    live_found = _discover_live_models(provider_tag, api_key, want_vision=bool(img_b64))
                if live_found:
                    candidate_models.extend(live_found)
                    attempt_cap += 1

    if is_byok:
        raise RuntimeError(f"BYOK_LIMIT: {last_error}")
    raise RuntimeError(last_error)


def _route_server_byok(
    user_keys: list,
    prompt_text: str,
    img_pil: Optional[Image.Image] = None,
    img_b64: Optional[str] = None,
    deep: bool = False,
    sys_prompt: str = PROMPT,
) -> str:
    grouped = {k: [] for k in ("groq", "gemini", "nvidia", "openrouter", "cerebras", "mistral", "openai")}
    for raw_k in user_keys:
        prov, clean_k = detect_key_provider(raw_k)
        if prov in grouped:
            grouped[prov].append(clean_k)

    if not any(grouped.values()):
        raise RuntimeError("BYOK_INVALID: Invalid personal API key (Alt+K)")

    errors = []

    for prov in ("groq", "nvidia", "openrouter", "cerebras", "mistral"):
        meta = PROVIDER_META[prov]
        for api_key in grouped[prov]:
            try:
                return call_openai_compat(
                    endpoint=meta["chat_url"],
                    api_key=api_key,
                    models=meta["vision"] if img_b64 else meta["text"],
                    prompt_text=prompt_text,
                    img_b64=img_b64,
                    provider_tag=prov,
                    is_byok=True,
                    sys_prompt=sys_prompt,
                )
            except Exception as exc:
                errors.append(str(exc))

    if grouped["gemini"]:
        try:
            return call_gemini_keys(
                grouped["gemini"], prompt_text, img_pil=img_pil, deep=deep, is_byok=True, sys_prompt=sys_prompt
            )
        except Exception as exc:
            errors.append(str(exc))

    for oa_key in grouped["openai"]:
        try:
            return call_openai_compat(
                endpoint="https://api.openai.com/v1/chat/completions",
                api_key=oa_key,
                models=["gpt-4o-mini"],
                prompt_text=prompt_text,
                img_b64=img_b64,
                provider_tag="openai",
                is_byok=True,
                sys_prompt=sys_prompt,
            )
        except Exception as exc:
            errors.append(str(exc))

    if errors and all("BYOK_INVALID" in e for e in errors):
        raise RuntimeError("BYOK_INVALID: Invalid personal API key (Alt+K)")
    raise RuntimeError("BYOK_LIMIT: " + (" | ".join(errors) if errors else "Personal key busy"))


def _call_compat_pool(
    provider_tag: str,
    prompt_text: str,
    img_b64: Optional[str] = None,
    sys_prompt: str = PROMPT,
    max_models: int = 2,
) -> Optional[str]:
    keys = get_pool_keys(provider_tag)
    if not keys:
        return None
    meta = PROVIDER_META[provider_tag]
    start_idx = next(_KEY_CYCLE) % len(keys)
    ordered = keys[start_idx:] + keys[:start_idx]
    for api_key in ordered[:2]:
        try:
            res = call_openai_compat(
                endpoint=meta["chat_url"],
                api_key=api_key,
                models=meta["vision"] if img_b64 else meta["text"],
                prompt_text=prompt_text,
                img_b64=img_b64,
                provider_tag=provider_tag,
                is_byok=False,
                sys_prompt=sys_prompt,
                max_models=max_models,
            )
            if res:
                return res
        except Exception:
            continue
    return None


def _call_cloudflare_pool(prompt_text: str, img_b64: Optional[str] = None, sys_prompt: str = PROMPT) -> Optional[str]:
    account_id, keys, status = _get_cloudflare_cfg()
    if status != "ok" or not account_id or not keys:
        return None
    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
    models = CLOUDFLARE_VISION_MODELS if img_b64 else CLOUDFLARE_TEXT_MODELS
    for api_key in keys[:2]:
        try:
            res = call_openai_compat(
                endpoint=endpoint,
                api_key=api_key,
                models=models,
                prompt_text=prompt_text,
                img_b64=img_b64,
                provider_tag="cloudflare",
                is_byok=False,
                sys_prompt=sys_prompt,
                max_models=1,
            )
            if res:
                return res
        except Exception:
            continue
    return None


def _has_ready_gemini(deep: bool = False) -> bool:
    now = time.time()
    models = _get_live_or_default("gemini:live", GEMINI_DEEP_MODELS if deep else GEMINI_FAST_MODELS)
    for api_key in get_pool_keys("gemini"):
        if KEY_COOLDOWN.get(api_key, 0) > now:
            continue
        for model_name in models:
            if KEY_COOLDOWN.get(f"{api_key}:{model_name}", 0) <= now:
                return True
    return False


def _run_shared_cascade(
    contents_text: str,
    img_pil: Optional[Image.Image],
    img_b64_str: Optional[str],
    is_followup: bool,
    sys_prompt: str,
    hedged: bool = False,
) -> str:
    errors = []
    groq_tried = False
    gemini_tried = False
    gemini_ready = _has_ready_gemini(deep=is_followup)

    # Tier 1 (Sub-2s Fast Hedge): Groq lead <-> 1.2s staggered hedge to Gemini
    if hedged and get_pool_keys("groq"):
        done_event = threading.Event()
        wake_backup = threading.Event()

        def _groq_lead():
            try:
                res = _call_compat_pool("groq", contents_text, img_b64=img_b64_str, sys_prompt=sys_prompt, max_models=1)
                if res:
                    done_event.set()
                    wake_backup.set()
                    return res
                wake_backup.set()
                raise RuntimeError("Groq unavailable")
            except Exception:
                wake_backup.set()
                raise

        def _gemini_backup():
            wake_backup.wait(1.2)
            if done_event.is_set():
                raise RuntimeError("Cancelled")
            res = call_gemini_keys(
                keys=get_pool_keys("gemini"),
                prompt_text=contents_text,
                img_pil=img_pil,
                deep=is_followup,
                is_byok=False,
                sys_prompt=sys_prompt,
            )
            if res:
                done_event.set()
            return res

        futures = [_HEDGE_POOL.submit(_groq_lead)]
        groq_tried = True
        if gemini_ready:
            futures.append(_HEDGE_POOL.submit(_gemini_backup))
            gemini_tried = True

        for fut in as_completed(futures):
            try:
                out = fut.result()
                if out:
                    done_event.set()
                    wake_backup.set()
                    return out
            except Exception as exc:
                if str(exc) != "Cancelled":
                    errors.append(str(exc))
        done_event.set()
        wake_backup.set()

    # Ordered 5-Provider Zero-Card Fallbacks (capped to 1 model per fallback so latency never snowballs)
    if not groq_tried:
        raw_groq = _call_compat_pool("groq", contents_text, img_b64=img_b64_str, sys_prompt=sys_prompt, max_models=2)
        if raw_groq:
            return raw_groq

    if not gemini_tried and gemini_ready:
        try:
            return call_gemini_keys(
                keys=get_pool_keys("gemini"),
                prompt_text=contents_text,
                img_pil=img_pil,
                deep=is_followup,
                is_byok=False,
                sys_prompt=sys_prompt,
            )
        except Exception as exc:
            errors.append(f"Gemini: {exc}")

    raw_cf = _call_cloudflare_pool(contents_text, img_b64=img_b64_str, sys_prompt=sys_prompt)
    if raw_cf:
        return raw_cf

    raw_nv = _call_compat_pool("nvidia", contents_text, img_b64=img_b64_str, sys_prompt=sys_prompt, max_models=1)
    if raw_nv:
        return raw_nv

    raw_or = _call_compat_pool("openrouter", contents_text, img_b64=img_b64_str, sys_prompt=sys_prompt, max_models=1)
    if raw_or:
        return raw_or

    raise RuntimeError(" | ".join(errors) or "All 5 providers temporarily busy")


def _unified_execute(
    req: AnalyzeRequest,
    action_mode: str = "solve",
    preserve_notes: bool = False,
    high_res_vision: bool = False,
    hedged: bool = False,
    custom_prompt: Optional[str] = None,
    device_id: Optional[str] = None,
    ip_hash: str = "",
) -> dict:
    if req.image_base64 and len(req.image_base64) > MAX_IMAGE_B64_CHARS:
        return {
            "detected_type": "ERROR",
            "cursor_pill": "Image too large — crop smaller area",
            "copy_value": "",
            "expanded_note": "",
            "pointer_xy": None,
            "key_source": "POOL",
        }
    if req.text_content and len(req.text_content) > MAX_TEXT_CHARS:
        req.text_content = req.text_content[:MAX_TEXT_CHARS]

    mode = (action_mode or "solve").strip().lower()
    if custom_prompt and custom_prompt.strip():
        sys_prompt = custom_prompt.strip()[:3000]
    elif mode == "rewrite":
        sys_prompt = V3_REWRITE_PROMPT
    elif mode == "explain":
        sys_prompt = V3_EXPLAIN_PROMPT
    else:
        sys_prompt = V3_SOLVE_PROMPT if preserve_notes else PROMPT

    user_keys = parse_user_keys(req.user_api_key)
    is_byok = bool(user_keys)
    with _COMMUNITY_LOCK:
        is_pro_pool = bool(device_id and device_id in _PRO_DEVICES and not is_byok)

    txt = (req.text_content or "").strip()
    if txt.upper().startswith("TEST:") or (req.user_api_key or "").strip().lower() == "test":
        label = txt[5:].strip() or "Verified UI & Hotkey Check"
        return {
            "detected_type": "DIRECT_VALUE",
            "cursor_pill": f"✓ {label[:55]}",
            "copy_value": label,
            "expanded_note": "Zero-token test mode active — 0 API credits spent.",
            "pointer_xy": None,
            "key_source": "TEST",
        }

    contents_text = ""
    img_pil = None
    img_b64_str = None
    raw_key = ""
    quiz_hash = ""
    is_followup = bool(req.follow_up and req.prior_context)

    if is_followup:
        contents_text = f"Context:\n{req.prior_context}\n\nFollow-up:\n{req.follow_up}"
        raw_key = f"fu:{contents_text}"
    elif req.image_base64:
        clean_b64 = req.image_base64.split(",")[-1]
        img_pil = Image.open(BytesIO(base64.b64decode(clean_b64))).convert("RGB")
        # Exact MD5 only — never collides across different questions
        raw_key = f"{mode}:img:{hashlib.md5(clean_b64.encode()).hexdigest()}"

        max_dim = 780 if high_res_vision else 700
        img_pil.thumbnail((max_dim, max_dim), Image.LANCZOS)

        small_img = img_pil.copy()
        compat_dim = 760 if high_res_vision else 512
        small_img.thumbnail((compat_dim, compat_dim), Image.BILINEAR)
        buffer = BytesIO()
        small_img.save(buffer, format="JPEG", quality=74 if high_res_vision else 68)
        img_b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        contents_text = (
            "Rewrite or reply to the highlighted message shown in this image."
            if mode == "rewrite"
            else "Explain the concept, code, or error shown in this image."
            if mode == "explain"
            else "Solve the primary problem shown in this image accurately."
        )
    elif req.text_content:
        contents_text = f"Solve:\n{req.text_content}" if mode == "solve" else f"Input:\n{req.text_content}"
        raw_key = f"{mode}:txt:{req.text_content}"
        if mode == "solve" and not custom_prompt:
            quiz_hash = normalize_quiz_hash(req.text_content)
    else:
        return {
            "detected_type": "ERROR",
            "cursor_pill": "Select text or snip area first",
            "copy_value": "",
            "expanded_note": "",
            "pointer_xy": None,
            "key_source": "BYOK" if is_byok else "POOL",
        }

    cache_key = hashlib.md5(f"v13:{raw_key}".encode()).hexdigest()

    if not is_followup:
        hit = IMAGE_CACHE.get(cache_key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            cached = dict(hit[1])
            if not preserve_notes:
                cached["expanded_note"] = ""
            cached["key_source"] = "BYOK" if is_byok else "CACHE"
            return cached

    if not is_followup and quiz_hash:
        q_hit = QUIZ_CACHE.get(quiz_hash)
        if q_hit and (time.time() - q_hit[0] < QUIZ_CACHE_TTL):
            cached = dict(q_hit[1])
            if not preserve_notes:
                cached["expanded_note"] = ""
            cached["key_source"] = "BYOK" if is_byok else "CACHE"
            return cached

        db_hit = _db_get(quiz_hash)
        if db_hit:
            QUIZ_CACHE[quiz_hash] = (time.time(), db_hit)
            cached = dict(db_hit)
            if not preserve_notes:
                cached["expanded_note"] = ""
            cached["key_source"] = "BYOK" if is_byok else "CACHE"
            return cached

    if is_byok:
        try:
            raw_json = _route_server_byok(
                user_keys=user_keys,
                prompt_text=contents_text,
                img_pil=img_pil,
                img_b64=img_b64_str,
                deep=is_followup,
                sys_prompt=sys_prompt,
            )
        except Exception as exc:
            err_msg = str(exc)
            if err_msg.startswith("BYOK_INVALID"):
                return {
                    "detected_type": "ERROR",
                    "cursor_pill": "Invalid or expired personal key (Alt+K)",
                    "copy_value": "",
                    "expanded_note": "",
                    "pointer_xy": None,
                    "key_source": "BYOK",
                }
            return {
                "detected_type": "ERROR",
                "cursor_pill": "Personal key rate-limited — retry",
                "copy_value": "",
                "expanded_note": err_msg[:200],
                "pointer_xy": None,
                "key_source": "BYOK",
            }
    else:
        limit_status = _check_all_limits_pipelined(device_id, ip_hash, is_pro_pool)
        if limit_status == "burst":
            return {
                "detected_type": "ERROR",
                "cursor_pill": "Slow down — wait a few seconds",
                "copy_value": "",
                "expanded_note": "Burst protection active (10 req/min). Add your own free key in Alt+K for unlimited speed.",
                "pointer_xy": None,
                "key_source": "POOL",
            }
        if limit_status == "quota":
            return {
                "detected_type": "ERROR",
                "cursor_pill": "Daily free limit reached — add free key in Alt+K",
                "copy_value": "",
                "expanded_note": "Paste your own free Groq/Gemini key in Alt+K, or type share:<KEY> to unlock 150/day Pro Pool.",
                "pointer_xy": None,
                "key_source": "POOL",
            }

        raw_json = _run_shared_cascade(
            contents_text=contents_text,
            img_pil=img_pil,
            img_b64_str=img_b64_str,
            is_followup=is_followup,
            sys_prompt=sys_prompt,
            hedged=hedged or is_pro_pool,
        )

    result = extract_json(raw_json, preserve_notes=True)
    confidence = result.pop("confidence", "high")
    result["key_source"] = "BYOK" if is_byok else ("PRO" if is_pro_pool else "POOL")

    if not re.search(r"unclear crop", result.get("cursor_pill", ""), re.I):
        IMAGE_CACHE[cache_key] = (time.time(), dict(result))
        if len(IMAGE_CACHE) > 600:
            IMAGE_CACHE.pop(next(iter(IMAGE_CACHE)))

        if not is_followup and quiz_hash and confidence == "high":
            QUIZ_CACHE[quiz_hash] = (time.time(), dict(result))
            if len(QUIZ_CACHE) > 900:
                QUIZ_CACHE.pop(next(iter(QUIZ_CACHE)))
            _HEDGE_POOL.submit(_db_put, quiz_hash, dict(result))

    if not preserve_notes and result.get("detected_type") in ("MCQ", "DIRECT_VALUE", "CODE_FIX"):
        result["expanded_note"] = ""

    return result


@app.post("/analyze")
def analyze_context(req: AnalyzeRequest, request: Request):
    try:
        return _unified_execute(
            req,
            action_mode="solve",
            preserve_notes=False,
            ip_hash=_get_client_ip_hash(request),
        )
    except Exception as exc:
        print(f"[BACKEND ERROR]: {exc}")
        return {
            "detected_type": "ERROR",
            "cursor_pill": "High traffic — press hotkey again",
            "copy_value": "",
            "expanded_note": "",
            "pointer_xy": None,
            "key_source": "POOL",
        }


@app.post("/v3/analyze")
def v3_analyze_context(req: V3AnalyzeRequest, request: Request):
    try:
        return _unified_execute(
            req,
            action_mode=req.action_mode or "solve",
            preserve_notes=True,
            device_id=req.device_id,
            ip_hash=_get_client_ip_hash(request),
        )
    except Exception as exc:
        print(f"[V3 BACKEND ERROR]: {exc}")
        return {
            "detected_type": "ERROR",
            "cursor_pill": "High traffic — press hotkey again",
            "copy_value": "",
            "expanded_note": str(exc)[:260],
            "pointer_xy": None,
            "key_source": "POOL",
        }


@app.post("/v3_1/analyze")
def v3_1_analyze_context(req: V3AnalyzeRequest, request: Request):
    try:
        return _unified_execute(
            req,
            action_mode=req.action_mode or "solve",
            preserve_notes=True,
            high_res_vision=True,
            hedged=True,
            custom_prompt=req.custom_prompt,
            device_id=req.device_id,
            ip_hash=_get_client_ip_hash(request),
        )
    except Exception as exc:
        print(f"[V3.1 BACKEND ERROR]: {exc}")
        return {
            "detected_type": "ERROR",
            "cursor_pill": "High traffic — press hotkey again",
            "copy_value": "",
            "expanded_note": str(exc)[:260],
            "pointer_xy": None,
            "key_source": "POOL",
        }

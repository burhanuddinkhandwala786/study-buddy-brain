import os
import re
import json
import base64
import time
import sqlite3
import hashlib
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Optional

from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import httpx
from google import genai
from google.genai import types

app = FastAPI(title="Cursor Companion Brain", version="5.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_HTTP_CLIENT = httpx.Client(timeout=10.5)
_HEDGE_POOL = ThreadPoolExecutor(max_workers=16)

IMAGE_CACHE = {}
CACHE_TTL = 900

QUIZ_CACHE = {}
QUIZ_CACHE_TTL = 3600

KEY_COOLDOWN = {}
_KEY_CYCLE = itertools.count()

_GEMINI_CLIENTS = {}
_GEMINI_CLIENTS_LOCK = threading.Lock()
_DB_LOCK = threading.Lock()
_DB_PATH = "/tmp/companion_brain_cache_v10.db"

GEMINI_FAST_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
]

GEMINI_DEEP_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

GROQ_TEXT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

GROQ_VISION_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
]

NVIDIA_TEXT_MODELS = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
]

NVIDIA_VISION_MODELS = [
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "mistralai/mistral-large-4",
]

OPENROUTER_TEXT_MODELS = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
]

OPENROUTER_VISION_MODELS = [
    "openrouter/free",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
]

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


def _init_db():
    try:
        with _DB_LOCK:
            conn = sqlite3.connect(_DB_PATH)
            conn.execute("CREATE TABLE IF NOT EXISTS qcache (qhash TEXT PRIMARY KEY, ts REAL, payload TEXT)")
            conn.commit()
            conn.close()
    except Exception:
        pass


_init_db()


def _db_get(qhash: str, max_age: int = 86400) -> Optional[dict]:
    if not qhash:
        return None
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
    try:
        with _DB_LOCK:
            conn = sqlite3.connect(_DB_PATH)
            conn.execute(
                "INSERT OR REPLACE INTO qcache (qhash, ts, payload) VALUES (?, ?, ?)",
                (qhash, time.time(), json.dumps(payload)),
            )
            conn.execute("DELETE FROM qcache WHERE ts < ?", (time.time() - 86400,))
            conn.commit()
            conn.close()
    except Exception:
        pass


def get_keys(env_name: str):
    return [
        key.strip().strip('"').strip("'")
        for key in os.environ.get(env_name, "").split(",")
        if key.strip().strip('"').strip("'")
    ]


def parse_user_keys(raw_user_keys: Optional[str]):
    cleaned_raw = re.sub(r"\b(?:bearer|api_key|key)\s*[:=]?\s*", "", raw_user_keys or "", flags=re.I)
    return [
        key.strip().strip('"').strip("'")
        for key in re.split(r"[\s,;]+", cleaned_raw)
        if key.strip().strip('"').strip("'")
    ]


def detect_key_provider(api_key: str) -> str:
    if api_key.startswith(("AIza", "AQ.")):
        return "gemini"
    if api_key.startswith("gsk_"):
        return "groq"
    if api_key.startswith("nvapi-"):
        return "nvidia"
    if api_key.startswith("sk-or-"):
        return "openrouter"
    if api_key.startswith("sk-"):
        return "openai"
    return "unknown"


def normalize_quiz_hash(text: str) -> str:
    s = re.sub(
        r"^\s*(?:q(?:uestion)?\s*\d+[\.\)\:\-]*|\d+[\.\)\:\-]+|\[\d+\s*marks?\])\s*",
        "",
        (text or "").strip(),
        flags=re.I,
    )
    words = re.findall(r"[a-z0-9]+", s.lower())
    cleaned = "".join(words)
    return hashlib.md5(f"v10:{cleaned}".encode()).hexdigest() if (len(cleaned) >= 35 and len(words) >= 5) else ""


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None
    user_api_key: Optional[str] = None


class V3AnalyzeRequest(AnalyzeRequest):
    action_mode: Optional[str] = "solve"
    device_id: Optional[str] = None


def _get_client(api_key: str):
    client = _GEMINI_CLIENTS.get(api_key)
    if client is None:
        with _GEMINI_CLIENTS_LOCK:
            client = _GEMINI_CLIENTS.get(api_key)
            if client is None:
                client = genai.Client(api_key=api_key, http_options={"timeout": 6000})
                _GEMINI_CLIENTS[api_key] = client
    return client


@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/healthz", methods=["GET", "HEAD"])
def health():
    now = time.time()
    return {
        "status": "ok",
        "engine": "v5.2.0-rocksolid-v31",
        "gemini_keys": len(get_keys("GEMINI_API_KEY")),
        "groq_keys": len(get_keys("GROQ_API_KEY")),
        "nvidia_keys": len(get_keys("NVIDIA_API_KEY")),
        "openrouter_keys": len(get_keys("OPENROUTER_API_KEY")),
        "cached_items": len(IMAGE_CACHE) + len(QUIZ_CACHE),
        "active_cooldowns": sum(1 for exp in KEY_COOLDOWN.values() if exp > now),
    }


@app.get("/diag")
def diagnose_keys():
    report = {"gemini": [], "groq": "not_configured", "nvidia": "not_configured", "openrouter": "not_configured"}

    for index, api_key in enumerate(get_keys("GEMINI_API_KEY"), 1):
        try:
            client = _get_client(api_key)
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
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

    for tag, env_name, url, model in [
        ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions", GROQ_TEXT_MODELS[0]),
        ("nvidia", "NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1/chat/completions", NVIDIA_TEXT_MODELS[0]),
        ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions", OPENROUTER_TEXT_MODELS[0]),
    ]:
        keys = get_keys(env_name)
        if keys:
            try:
                body = {"model": model, "messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 64}
                if tag == "groq" and "gpt-oss" in model:
                    body["reasoning_effort"] = "low"
                r = _HTTP_CLIENT.post(
                    url,
                    json=body,
                    headers={"Authorization": f"Bearer {keys[0]}", "Content-Type": "application/json"},
                )
                report[tag] = "HEALTHY" if r.status_code == 200 else f"HTTP {r.status_code}: {r.text[:100]}"
            except Exception as exc:
                report[tag] = f"ERROR: {str(exc)[:100]}"

    return report


@app.get("/selftest")
@app.get("/v3/selftest")
def self_test():
    results = {}
    t0 = time.time()

    try:
        raw_groq = call_openai_compat(
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            api_key=get_keys("GROQ_API_KEY")[0],
            models=GROQ_TEXT_MODELS,
            prompt_text='Return JSON: {"verification":"ok","detected_type":"DIRECT_VALUE","cursor_pill":"2","copy_value":"2","expanded_note":"","pointer_xy":null}',
            provider_tag="groq",
        )
        results["groq_pipeline"] = {"status": "200 OK", "output": extract_json(raw_groq).get("cursor_pill")}
    except Exception as exc:
        results["groq_pipeline"] = {"status": "FAIL", "error": str(exc)[:120]}

    try:
        raw_gemini = call_gemini_keys(
            keys=get_keys("GEMINI_API_KEY"),
            prompt_text='Return JSON: {"verification":"ok","detected_type":"DIRECT_VALUE","cursor_pill":"4","copy_value":"4","expanded_note":"","pointer_xy":null}',
            is_byok=False,
        )
        results["gemini_pipeline"] = {"status": "200 OK", "output": extract_json(raw_gemini).get("cursor_pill")}
    except Exception as exc:
        results["gemini_pipeline"] = {"status": "FAIL", "error": str(exc)[:120]}

    byok_guard = analyze_context(
        AnalyzeRequest(
            text_content="What is 2+2?",
            user_api_key="AIzaInvalidTestKey1, AIzaInvalidTestKey2",
        )
    )
    results["byok_isolation_guard"] = {
        "status": "200 OK" if byok_guard.get("key_source") == "BYOK" else "FAIL",
        "message": byok_guard.get("cursor_pill"),
    }

    mock_check = v3_analyze_context(V3AnalyzeRequest(text_content="TEST: UI alignment check"))
    results["zero_token_test_mode"] = {
        "status": "200 OK (0 tokens)" if mock_check.get("key_source") == "TEST" else "FAIL",
        "pill": mock_check.get("cursor_pill"),
    }
    results["elapsed_ms"] = round((time.time() - t0) * 1000)
    return results


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

    # Remove internal verification scratchpad so client payload stays lean
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
    if "429" in error_text or "resource_exhausted" in lowered or "quota" in lowered:
        return "rate_limit"
    if (
        "401" in error_text
        or "403" in error_text
        or "permission_denied" in lowered
        or "api_key_invalid" in lowered
        or "api key not valid" in lowered
    ):
        return "forbidden"
    if "404" in error_text or "not_found" in lowered or "decommission" in lowered:
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
    # Omit temperature on Gemini 3.x to prevent reasoning loops; cap max_output_tokens at 850 to protect free-tier quota
    cfg_kwargs = {
        "response_mime_type": "application/json",
        "max_output_tokens": 850,
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
    models = GEMINI_DEEP_MODELS if deep else GEMINI_FAST_MODELS
    last_error = "Gemini keys cooling down"

    for model_name in models:
        use_thinking = "lite" not in model_name
        for api_key in ordered_keys:
            if KEY_COOLDOWN.get(api_key, 0) > now:
                continue
            model_cd_key = f"{api_key}:{model_name}"
            if KEY_COOLDOWN.get(model_cd_key, 0) > now:
                continue

            try:
                return _try_one_gemini(
                    api_key, model_name, prompt_text, img_pil, use_thinking, sys_prompt=sys_prompt
                )
            except Exception as exc:
                error_text = str(exc)
                last_error = f"{model_name}: {error_text[:100]}"
                err_type = _classify_gemini_error(error_text)

                if err_type == "rate_limit":
                    KEY_COOLDOWN[model_cd_key] = time.time() + 25
                elif err_type == "forbidden":
                    KEY_COOLDOWN[api_key] = time.time() + 3600
                elif err_type == "model_unavailable":
                    KEY_COOLDOWN[model_cd_key] = time.time() + 3600

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
) -> str:
    last_error = f"No {provider_tag} attempt succeeded"

    for model_id in models:
        cooldown_key = f"{provider_tag}:{api_key}:{model_id}"
        if KEY_COOLDOWN.get(cooldown_key, 0) > time.time():
            continue

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
            "max_tokens": 950,
        }
        if provider_tag == "groq" and "gpt-oss" in model_id:
            payload_body["reasoning_effort"] = "medium"
        if provider_tag == "groq" and not img_b64:
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
        if response.status_code in (401, 403):
            KEY_COOLDOWN[f"{provider_tag}:{api_key}"] = time.time() + 3600
            if is_byok:
                raise RuntimeError("BYOK_INVALID: Invalid personal API key (Alt+K)")
            break
        if response.status_code == 429:
            KEY_COOLDOWN[cooldown_key] = time.time() + 25
        elif response.status_code in (400, 404, 410):
            KEY_COOLDOWN[cooldown_key] = time.time() + 86400

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
    gemini_keys = [k for k in user_keys if detect_key_provider(k) == "gemini"]
    groq_keys = [k for k in user_keys if detect_key_provider(k) == "groq"]
    nvidia_keys = [k for k in user_keys if detect_key_provider(k) == "nvidia"]
    openrouter_keys = [k for k in user_keys if detect_key_provider(k) == "openrouter"]
    openai_keys = [k for k in user_keys if detect_key_provider(k) == "openai"]

    if not (gemini_keys or groq_keys or nvidia_keys or openrouter_keys or openai_keys):
        raise RuntimeError("BYOK_INVALID: Invalid personal API key (Alt+K)")

    errors = []

    if gemini_keys:
        try:
            return call_gemini_keys(
                gemini_keys, prompt_text, img_pil=img_pil, deep=deep, is_byok=True, sys_prompt=sys_prompt
            )
        except Exception as exc:
            errors.append(str(exc))

    for g_key in groq_keys:
        try:
            return call_openai_compat(
                endpoint="https://api.groq.com/openai/v1/chat/completions",
                api_key=g_key,
                models=GROQ_VISION_MODELS if img_b64 else GROQ_TEXT_MODELS,
                prompt_text=prompt_text,
                img_b64=img_b64,
                provider_tag="groq",
                is_byok=True,
                sys_prompt=sys_prompt,
            )
        except Exception as exc:
            errors.append(str(exc))

    for nv_key in nvidia_keys:
        try:
            return call_openai_compat(
                endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
                api_key=nv_key,
                models=NVIDIA_VISION_MODELS if img_b64 else NVIDIA_TEXT_MODELS,
                prompt_text=prompt_text,
                img_b64=img_b64,
                provider_tag="nvidia",
                is_byok=True,
                sys_prompt=sys_prompt,
            )
        except Exception as exc:
            errors.append(str(exc))

    for or_key in openrouter_keys:
        try:
            return call_openai_compat(
                endpoint="https://openrouter.ai/api/v1/chat/completions",
                api_key=or_key,
                models=OPENROUTER_VISION_MODELS if img_b64 else OPENROUTER_TEXT_MODELS,
                prompt_text=prompt_text,
                img_b64=img_b64,
                provider_tag="openrouter",
                is_byok=True,
                sys_prompt=sys_prompt,
            )
        except Exception as exc:
            errors.append(str(exc))

    for oa_key in openai_keys:
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


def _call_groq_pool(prompt_text: str, img_b64: Optional[str] = None, sys_prompt: str = PROMPT) -> Optional[str]:
    for g_key in get_keys("GROQ_API_KEY"):
        try:
            res = call_openai_compat(
                endpoint="https://api.groq.com/openai/v1/chat/completions",
                api_key=g_key,
                models=GROQ_VISION_MODELS if img_b64 else GROQ_TEXT_MODELS,
                prompt_text=prompt_text,
                img_b64=img_b64,
                provider_tag="groq",
                is_byok=False,
                sys_prompt=sys_prompt,
            )
            if res:
                return res
        except Exception:
            continue
    return None


def _call_nvidia_pool(prompt_text: str, img_b64: Optional[str] = None, sys_prompt: str = PROMPT) -> Optional[str]:
    for nv_key in get_keys("NVIDIA_API_KEY"):
        try:
            res = call_openai_compat(
                endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
                api_key=nv_key,
                models=NVIDIA_VISION_MODELS if img_b64 else NVIDIA_TEXT_MODELS,
                prompt_text=prompt_text,
                img_b64=img_b64,
                provider_tag="nvidia",
                is_byok=False,
                sys_prompt=sys_prompt,
            )
            if res:
                return res
        except Exception:
            continue
    return None


def _call_openrouter_pool(prompt_text: str, img_b64: Optional[str] = None, sys_prompt: str = PROMPT) -> Optional[str]:
    for or_key in get_keys("OPENROUTER_API_KEY"):
        try:
            res = call_openai_compat(
                endpoint="https://openrouter.ai/api/v1/chat/completions",
                api_key=or_key,
                models=OPENROUTER_VISION_MODELS if img_b64 else OPENROUTER_TEXT_MODELS,
                prompt_text=prompt_text,
                img_b64=img_b64,
                provider_tag="openrouter",
                is_byok=False,
                sys_prompt=sys_prompt,
            )
            if res:
                return res
        except Exception:
            continue
    return None


def _run_shared_cascade(
    contents_text: str,
    img_pil: Optional[Image.Image],
    img_b64_str: Optional[str],
    is_followup: bool,
    sys_prompt: str,
    hedged: bool = False,
) -> str:
    if hedged:
        done_event = threading.Event()

        def _gemini_branch():
            return call_gemini_keys(
                keys=get_keys("GEMINI_API_KEY"),
                prompt_text=contents_text,
                img_pil=img_pil,
                deep=is_followup,
                is_byok=False,
                sys_prompt=sys_prompt,
            )

        def _groq_staggered_branch():
            # Wait 1.6s so Gemini handles ~90% of calls solo without burning Groq free-tier tokens unless Gemini lags
            if done_event.wait(1.6):
                raise RuntimeError("Cancelled")
            res = _call_groq_pool(contents_text, img_b64=img_b64_str, sys_prompt=sys_prompt)
            if not res:
                raise RuntimeError("Groq unavailable")
            return res

        futures = [
            _HEDGE_POOL.submit(_gemini_branch),
            _HEDGE_POOL.submit(_groq_staggered_branch),
        ]
        for fut in as_completed(futures):
            try:
                out = fut.result()
                if out:
                    done_event.set()
                    return out
            except Exception:
                continue
        done_event.set()

    errors = []
    try:
        return call_gemini_keys(
            keys=get_keys("GEMINI_API_KEY"),
            prompt_text=contents_text,
            img_pil=img_pil,
            deep=is_followup,
            is_byok=False,
            sys_prompt=sys_prompt,
        )
    except Exception as exc:
        errors.append(f"Gemini: {exc}")

    raw_groq = _call_groq_pool(contents_text, img_b64=img_b64_str, sys_prompt=sys_prompt)
    if raw_groq:
        return raw_groq

    raw_nv = _call_nvidia_pool(contents_text, img_b64=img_b64_str, sys_prompt=sys_prompt)
    if raw_nv:
        return raw_nv

    raw_or = _call_openrouter_pool(contents_text, img_b64=img_b64_str, sys_prompt=sys_prompt)
    if raw_or:
        return raw_or

    raise RuntimeError(" | ".join(errors) or "All providers busy")


def _unified_execute(
    req: AnalyzeRequest,
    action_mode: str = "solve",
    preserve_notes: bool = False,
    high_res_vision: bool = False,
    hedged: bool = False,
) -> dict:
    mode = (action_mode or "solve").strip().lower()
    sys_prompt = (
        V3_REWRITE_PROMPT
        if mode == "rewrite"
        else V3_EXPLAIN_PROMPT
        if mode == "explain"
        else (V3_SOLVE_PROMPT if preserve_notes else PROMPT)
    )

    user_keys = parse_user_keys(req.user_api_key)
    is_byok = bool(user_keys)

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
        raw_key = f"{mode}:img:{hashlib.md5(clean_b64.encode()).hexdigest()}"

        img_pil = Image.open(BytesIO(base64.b64decode(clean_b64))).convert("RGB")
        # 800px @ q76 keeps small exam text sharp while cutting vision upload latency & token spend
        max_dim = 800 if high_res_vision else 720
        img_pil.thumbnail((max_dim, max_dim), Image.LANCZOS)

        small_img = img_pil.copy()
        compat_dim = 768 if high_res_vision else 480
        small_img.thumbnail((compat_dim, compat_dim), Image.BILINEAR)
        buffer = BytesIO()
        small_img.save(buffer, format="JPEG", quality=76 if high_res_vision else 68)
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
        if mode == "solve":
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

    cache_key = hashlib.md5(f"v10:{raw_key}".encode()).hexdigest()
    image_hit = IMAGE_CACHE.get(cache_key)
    if not is_followup and image_hit and time.time() - image_hit[0] < CACHE_TTL:
        cached = dict(image_hit[1])
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
                    "cursor_pill": "Invalid personal API key (Alt+K)",
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
        raw_json = _run_shared_cascade(
            contents_text=contents_text,
            img_pil=img_pil,
            img_b64_str=img_b64_str,
            is_followup=is_followup,
            sys_prompt=sys_prompt,
            hedged=hedged,
        )

    result = extract_json(raw_json, preserve_notes=True)
    confidence = result.pop("confidence", "high")
    result["key_source"] = "BYOK" if is_byok else "POOL"

    if not re.search(r"unclear crop", result.get("cursor_pill", ""), re.I):
        IMAGE_CACHE[cache_key] = (time.time(), dict(result))
        if len(IMAGE_CACHE) > 500:
            IMAGE_CACHE.pop(next(iter(IMAGE_CACHE)))

        if not is_followup and quiz_hash and confidence == "high":
            QUIZ_CACHE[quiz_hash] = (time.time(), dict(result))
            if len(QUIZ_CACHE) > 800:
                QUIZ_CACHE.pop(next(iter(QUIZ_CACHE)))
            _db_put(quiz_hash, dict(result))

    if not preserve_notes and result.get("detected_type") in ("MCQ", "DIRECT_VALUE", "CODE_FIX"):
        result["expanded_note"] = ""

    return result


@app.post("/analyze")
def analyze_context(req: AnalyzeRequest):
    try:
        return _unified_execute(req, action_mode="solve", preserve_notes=False)
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
def v3_analyze_context(req: V3AnalyzeRequest):
    try:
        return _unified_execute(req, action_mode=req.action_mode or "solve", preserve_notes=True)
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
def v3_1_analyze_context(req: V3AnalyzeRequest):
    try:
        return _unified_execute(
            req,
            action_mode=req.action_mode or "solve",
            preserve_notes=True,
            high_res_vision=True,
            hedged=True,
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

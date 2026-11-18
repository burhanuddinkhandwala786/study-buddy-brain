import os
import re
import json
import base64
import time
import sqlite3
import hashlib
import itertools
import threading
from io import BytesIO
from typing import Optional

from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import httpx
from google import genai
from google.genai import types

app = FastAPI(title="Cursor Companion Brain", version="4.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_HTTP_CLIENT = httpx.Client(timeout=10.0)

IMAGE_CACHE = {}
CACHE_TTL = 900  # 15-minute exact screenshot/payload dedup

QUIZ_CACHE = {}
QUIZ_CACHE_TTL = 1800  # 30-minute normalized question dedup across V1, V2 & V3

KEY_COOLDOWN = {}
_KEY_CYCLE = itertools.count()

_GEMINI_CLIENTS = {}
_GEMINI_CLIENTS_LOCK = threading.Lock()
_DB_LOCK = threading.Lock()
_DB_PATH = "/tmp/companion_brain_cache.db"

# Distinct Gemini model buckets so 3 keys = 9 independent daily/minute quota pools
GEMINI_FAST_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

GEMINI_VISION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

GROQ_TEXT_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]

GROQ_VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3.8-27b",
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
    "nvidia/nemotron-3.5-super-vl:free",
]

PROMPT = """You are an elite academic, competitive-exam, and engineering verification engine.
Solve step-by-step internally before outputting JSON:
1. If multiple questions appear in a screenshot, focus ONLY on the primary/centered question.
2. Read the ENTIRE question carefully (watch for NOT, INCORRECT, EXCEPT, and unit conversions).
3. Independently calculate or verify the exact factual answer BEFORE looking at any pre-selected radio buttons or highlights.
4. Double-check your math/logic, then match your verified answer to the exact option letter/number and text.

Return ONLY valid JSON:
{"detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"","pointer_xy":null}

Strict Output Rules:
- IF ANY OPTIONS (A/B/C/D or 1/2/3/4) ARE PRESENT: You MUST set detected_type="MCQ", cursor_pill = exact option letter + text (e.g. "C. A is correct R is wrong"), copy_value="", expanded_note="". NEVER output a long paragraph when options exist.
- DIRECT_VALUE / CODE_FIX (no options present): cursor_pill = concise verified answer or fixed code line; copy_value = raw answer; expanded_note="".
- CONCEPT_EXPLAIN (only when user explicitly asks "explain" or "why" in follow-up): cursor_pill = 1-line summary; expanded_note = 2 short bullet points."""

V3_SOLVE_PROMPT = """You are an elite academic, competitive-exam, and engineering verification engine.
Solve step-by-step internally before outputting JSON:
1. If multiple questions appear in a screenshot, focus ONLY on the primary/centered question.
2. Read the ENTIRE question carefully (watch for NOT, INCORRECT, EXCEPT, and unit conversions).
3. Independently calculate or verify the exact factual answer BEFORE looking at any pre-selected radio buttons or highlights.
4. Match your verified answer to the exact option letter/number and text.

Return ONLY valid JSON:
{"detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"...","confidence":"high"|"medium","pointer_xy":null}

Rules:
- IF ANY OPTIONS (A/B/C/D or 1/2/3/4) ARE PRESENT: detected_type="MCQ", cursor_pill = exact option letter + text (e.g. "C. A is correct R is wrong"), copy_value="", expanded_note = 1-sentence reason for sidecar history.
- DIRECT_VALUE / CODE_FIX: cursor_pill = exact verified answer or fixed code line; copy_value = same raw value; expanded_note = 1-sentence verification note.
- CONCEPT_EXPLAIN: cursor_pill = crisp 1-line takeaway; copy_value = takeaway; expanded_note = 3 concise bullet points."""

V3_REWRITE_PROMPT = """You are a sharp, natural executive writing assistant.
Rewrite or craft a crisp, polite, professional reply for the provided text without fluff.
Return ONLY valid JSON:
{"detected_type":"REWRITE","cursor_pill":"...","copy_value":"...","expanded_note":"...","confidence":"high","pointer_xy":null}
Rules:
- cursor_pill = concise 1-line preview of the polished rewrite (max 70 chars).
- copy_value = complete ready-to-paste polished text.
- expanded_note = brief note on tone improvement."""

V3_EXPLAIN_PROMPT = """You are an expert technical and academic tutor.
Explain the provided concept, code snippet, or error clearly in plain English.
Return ONLY valid JSON:
{"detected_type":"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"...","confidence":"high","pointer_xy":null}
Rules:
- cursor_pill = single crisp 1-line summary (max 75 chars).
- copy_value = same 1-line summary.
- expanded_note = 3 high-yield bullet points for the study sidecar."""


def _init_db():
    try:
        with _DB_LOCK:
            conn = sqlite3.connect(_DB_PATH)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS qcache (qhash TEXT PRIMARY KEY, ts REAL, payload TEXT)"
            )
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
    return [
        key.strip().strip('"').strip("'")
        for key in (raw_user_keys or "").split(",")
        if key.strip().strip('"').strip("'")
    ]


def detect_key_provider(api_key: str) -> str:
    if api_key.startswith("AIza"):
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
    # Strip leading question numbers (e.g., "Q12.", "12)", "[2 Marks]") so identical questions match across users
    s = re.sub(r"^\s*(?:q(?:uestion)?\s*\d+[\.\)\:\-]*|\d+[\.\)\:\-]+|\[\d+\s*marks?\])\s*", "", (text or "").strip(), flags=re.I)
    cleaned = re.sub(r"[^a-z0-9]+", "", s.lower())
    return hashlib.md5(cleaned.encode()).hexdigest() if len(cleaned) >= 22 else ""


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
                client = genai.Client(
                    api_key=api_key,
                    http_options={"timeout": 12000},
                )
                _GEMINI_CLIENTS[api_key] = client
    return client


@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/healthz", methods=["GET", "HEAD"])
def health():
    now = time.time()
    return {
        "status": "ok",
        "engine": "v4.4-quota-multiplier",
        "gemini_keys": len(get_keys("GEMINI_API_KEY")),
        "groq_keys": len(get_keys("GROQ_API_KEY")),
        "nvidia_keys": len(get_keys("NVIDIA_API_KEY")),
        "openrouter_keys": len(get_keys("OPENROUTER_API_KEY")),
        "cached_items": len(IMAGE_CACHE) + len(QUIZ_CACHE),
        "active_cooldowns": sum(1 for exp in KEY_COOLDOWN.values() if exp > now),
    }


@app.get("/diag")
def diagnose_keys():
    report = {
        "gemini": [],
        "groq": "not_configured",
        "nvidia": "not_configured",
        "openrouter": "not_configured",
    }

    for index, api_key in enumerate(get_keys("GEMINI_API_KEY"), 1):
        try:
            client = _get_client(api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents="Reply OK",
                config=types.GenerateContentConfig(max_output_tokens=10),
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

    groq_keys = get_keys("GROQ_API_KEY")
    if groq_keys:
        try:
            response = _HTTP_CLIENT.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": GROQ_TEXT_MODELS[0],
                    "messages": [{"role": "user", "content": "Reply OK"}],
                    "max_tokens": 10,
                },
                headers={
                    "Authorization": f"Bearer {groq_keys[0]}",
                    "Content-Type": "application/json",
                },
            )
            report["groq"] = (
                "HEALTHY"
                if response.status_code == 200
                else f"HTTP {response.status_code}: {response.text[:120]}"
            )
        except Exception as exc:
            report["groq"] = f"ERROR: {str(exc)[:120]}"

    nv_keys = get_keys("NVIDIA_API_KEY")
    if nv_keys:
        try:
            response = _HTTP_CLIENT.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                json={
                    "model": NVIDIA_TEXT_MODELS[0],
                    "messages": [{"role": "user", "content": "Reply OK"}],
                    "max_tokens": 10,
                },
                headers={
                    "Authorization": f"Bearer {nv_keys[0]}",
                    "Content-Type": "application/json",
                },
            )
            report["nvidia"] = (
                "HEALTHY"
                if response.status_code == 200
                else f"HTTP {response.status_code}: {response.text[:120]}"
            )
        except Exception as exc:
            report["nvidia"] = f"ERROR: {str(exc)[:120]}"

    or_keys = get_keys("OPENROUTER_API_KEY")
    if or_keys:
        try:
            response = _HTTP_CLIENT.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": OPENROUTER_TEXT_MODELS[0],
                    "messages": [{"role": "user", "content": "Reply OK"}],
                    "max_tokens": 10,
                },
                headers={
                    "Authorization": f"Bearer {or_keys[0]}",
                    "Content-Type": "application/json",
                },
            )
            report["openrouter"] = (
                "HEALTHY"
                if response.status_code == 200
                else f"HTTP {response.status_code}: {response.text[:120]}"
            )
        except Exception as exc:
            report["openrouter"] = f"ERROR: {str(exc)[:120]}"

    return report


@app.get("/selftest")
def self_test():
    results = {}
    start_time = time.time()

    try:
        raw_groq = call_openai_compat(
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            api_key=get_keys("GROQ_API_KEY")[0],
            models=GROQ_TEXT_MODELS,
            prompt_text='Return JSON: {"detected_type":"DIRECT_VALUE","cursor_pill":"2","copy_value":"2","expanded_note":"","pointer_xy":null}',
            provider_tag="groq",
        )
        parsed_groq = extract_json(raw_groq)
        results["groq_pipeline"] = {"status": "200 OK", "output": parsed_groq.get("cursor_pill")}
    except Exception as exc:
        results["groq_pipeline"] = {"status": "FAIL", "error": str(exc)[:140]}

    try:
        raw_gemini = call_gemini_keys(
            keys=get_keys("GEMINI_API_KEY"),
            prompt_text='Return JSON: {"detected_type":"DIRECT_VALUE","cursor_pill":"4","copy_value":"4","expanded_note":"","pointer_xy":null}',
            is_byok=False,
        )
        parsed_gemini = extract_json(raw_gemini)
        results["gemini_pipeline"] = {"status": "200 OK", "output": parsed_gemini.get("cursor_pill")}
    except Exception as exc:
        results["gemini_pipeline"] = {"status": "FAIL", "error": str(exc)[:140]}

    mock_check = v3_analyze_context(V3AnalyzeRequest(text_content="TEST: UI alignment check"))
    results["zero_token_test_mode"] = {
        "status": "200 OK (0 tokens)" if mock_check.get("key_source") == "TEST" else "FAIL",
        "pill": mock_check.get("cursor_pill"),
    }

    results["elapsed_ms"] = round((time.time() - start_time) * 1000)
    return results


@app.get("/v3/selftest")
def v3_self_test():
    return self_test()


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
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        pass

    if data is None:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(cleaned):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(cleaned[idx:])
                    if isinstance(obj, dict) and ("cursor_pill" in obj or "detected_type" in obj):
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
            data = {
                "detected_type": type_match.group(1) if type_match else "DIRECT_VALUE",
                "cursor_pill": pill_val,
                "copy_value": copy_match.group(1) if copy_match else pill_val,
                "expanded_note": note_match.group(1) if note_match else "",
                "pointer_xy": None,
            }

    if not isinstance(data, dict):
        lines = [
            ln.strip().lstrip("*-•").strip()
            for ln in cleaned.splitlines()
            if ln.strip() and not ln.strip().startswith("{") and not ln.strip().startswith("}")
        ]
        if not lines:
            raise ValueError("Could not parse response")
        first_line = lines[0][:85]
        data = {
            "detected_type": "DIRECT_VALUE",
            "cursor_pill": first_line,
            "copy_value": first_line,
            "expanded_note": " ".join(lines[:3])[:260] if preserve_notes else "",
            "pointer_xy": None,
        }

    data.setdefault("detected_type", "DIRECT_VALUE")
    data.setdefault("cursor_pill", "Done")
    data.setdefault("copy_value", "")
    data.setdefault("expanded_note", "")
    data.setdefault("pointer_xy", None)

    if not preserve_notes and data.get("detected_type") in ("MCQ", "DIRECT_VALUE", "CODE_FIX"):
        data["expanded_note"] = ""

    return data


def _classify_gemini_error(error_text: str) -> str:
    lowered = error_text.lower()
    if "429" in error_text or "resource_exhausted" in lowered or "quota" in lowered:
        return "rate_limit"
    if (
        "403" in error_text
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
    cfg_kwargs = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
        "max_output_tokens": 850 if use_thinking else 500,
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
    prefer_fast: bool = False,
) -> str:
    if not keys:
        raise RuntimeError("No Gemini keys configured")

    now = time.time()
    start_idx = next(_KEY_CYCLE) % len(keys)
    ordered_keys = keys[start_idx:] + keys[:start_idx]
    models = GEMINI_FAST_MODELS if prefer_fast else (GEMINI_VISION_MODELS if (deep or img_pil is not None) else GEMINI_FAST_MODELS)
    last_error = "Gemini keys cooling down"

    for model_name in models:
        use_thinking = (not prefer_fast) and (deep or img_pil is not None) and "lite" not in model_name
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
                last_error = f"{model_name}: {error_text[:120]}"
                err_type = _classify_gemini_error(error_text)

                if err_type == "rate_limit":
                    # Only cool down this specific key:model bucket so the key's other model buckets stay active
                    KEY_COOLDOWN[model_cd_key] = time.time() + 35
                elif err_type == "forbidden":
                    KEY_COOLDOWN[api_key] = time.time() + 3600
                elif err_type == "model_unavailable":
                    KEY_COOLDOWN[model_cd_key] = time.time() + 3600

    if is_byok and all(KEY_COOLDOWN.get(k, 0) > time.time() for k in keys):
        raise RuntimeError("BYOK_INVALID")
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

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if provider_tag == "openrouter":
            headers["HTTP-Referer"] = "https://study-buddy-brain.onrender.com"
            headers["X-Title"] = "Cursor Companion"

        payload_body = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 600,
        }
        if provider_tag == "groq" and not img_b64:
            payload_body["response_format"] = {"type": "json_object"}

        try:
            response = _HTTP_CLIENT.post(endpoint, json=payload_body, headers=headers)
        except Exception as exc:
            last_error = f"{provider_tag} {model_id}: {str(exc)[:100]}"
            continue

        if response.status_code == 200:
            try:
                content = response.json()["choices"][0]["message"]["content"]
                if content:
                    extract_json(content, preserve_notes=True)
                    return content
            except Exception as parse_exc:
                last_error = f"{provider_tag} {model_id} parse error: {str(parse_exc)[:80]}"
                continue

        if response.status_code == 400 and "response_format" in payload_body:
            payload_body.pop("response_format", None)
            try:
                retry_res = _HTTP_CLIENT.post(endpoint, json=payload_body, headers=headers)
                if retry_res.status_code == 200:
                    content = retry_res.json()["choices"][0]["message"]["content"]
                    if content:
                        extract_json(content, preserve_notes=True)
                        return content
            except Exception:
                pass

        last_error = f"{provider_tag} {model_id}: HTTP {response.status_code}"
        if response.status_code in (401, 403):
            KEY_COOLDOWN[f"{provider_tag}:{api_key}"] = time.time() + 3600
            if is_byok:
                raise RuntimeError("BYOK_INVALID")
            break
        if response.status_code == 429:
            KEY_COOLDOWN[cooldown_key] = time.time() + 30
        elif response.status_code in (400, 404, 410):
            KEY_COOLDOWN[cooldown_key] = time.time() + 86400

    if is_byok:
        raise RuntimeError(f"BYOK_LIMIT: {last_error}")
    raise RuntimeError(last_error)


def route_universal_byok(
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
        raise RuntimeError("BYOK_INVALID")

    errors = []

    if gemini_keys:
        try:
            return call_gemini_keys(
                gemini_keys, prompt_text, img_pil, deep=deep, is_byok=True, sys_prompt=sys_prompt, prefer_fast=True
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
        raise RuntimeError("BYOK_INVALID")
    raise RuntimeError(" | ".join(errors) if errors else "BYOK_LIMIT")


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
) -> str:
    errors = []
    is_text_only = img_pil is None

    if is_text_only:
        # Text Quota Shield: Groq -> Gemini Flash-Lite -> NVIDIA NIM -> OpenRouter Free
        for g_key in get_keys("GROQ_API_KEY"):
            try:
                raw_json = call_openai_compat(
                    endpoint="https://api.groq.com/openai/v1/chat/completions",
                    api_key=g_key,
                    models=GROQ_TEXT_MODELS,
                    prompt_text=contents_text,
                    provider_tag="groq",
                    is_byok=False,
                    sys_prompt=sys_prompt,
                )
                if raw_json:
                    return raw_json
            except Exception as exc:
                errors.append(f"Groq: {exc}")

        try:
            return call_gemini_keys(
                keys=get_keys("GEMINI_API_KEY"),
                prompt_text=contents_text,
                deep=is_followup,
                is_byok=False,
                sys_prompt=sys_prompt,
                prefer_fast=True,
            )
        except Exception as exc:
            errors.append(f"Gemini: {exc}")

        raw_nv = _call_nvidia_pool(contents_text, sys_prompt=sys_prompt)
        if raw_nv:
            return raw_nv

        raw_or = _call_openrouter_pool(contents_text, sys_prompt=sys_prompt)
        if raw_or:
            return raw_or
    else:
        # Vision Quota Shield: Gemini Vision -> Groq Vision -> NVIDIA Vision -> OpenRouter Free Vision
        try:
            return call_gemini_keys(
                keys=get_keys("GEMINI_API_KEY"),
                prompt_text=contents_text,
                img_pil=img_pil,
                deep=is_followup,
                is_byok=False,
                sys_prompt=sys_prompt,
                prefer_fast=False,
            )
        except Exception as exc:
            errors.append(f"Gemini: {exc}")

        for g_key in get_keys("GROQ_API_KEY"):
            try:
                raw_json = call_openai_compat(
                    endpoint="https://api.groq.com/openai/v1/chat/completions",
                    api_key=g_key,
                    models=GROQ_VISION_MODELS,
                    prompt_text=contents_text,
                    img_b64=img_b64_str,
                    provider_tag="groq",
                    is_byok=False,
                    sys_prompt=sys_prompt,
                )
                if raw_json:
                    return raw_json
            except Exception as exc:
                errors.append(f"Groq: {exc}")

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
) -> dict:
    mode = (action_mode or "solve").strip().lower()
    sys_prompt = (
        V3_REWRITE_PROMPT
        if mode == "rewrite"
        else V3_EXPLAIN_PROMPT
        if mode == "explain"
        else (V3_SOLVE_PROMPT if preserve_notes else PROMPT)
    )

    # Zero-token TEST mode so you can test UI, hotkeys, and builds without burning API quota
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
    user_keys = parse_user_keys(req.user_api_key)
    is_byok = bool(user_keys)

    if is_followup:
        contents_text = f"Context:\n{req.prior_context}\n\nFollow-up:\n{req.follow_up}"
        raw_key = f"fu:{contents_text}"
    elif req.image_base64:
        clean_b64 = req.image_base64.split(",")[-1]
        raw_key = f"{mode}:img:{hashlib.md5(clean_b64.encode()).hexdigest()}"

        img_pil = Image.open(BytesIO(base64.b64decode(clean_b64))).convert("RGB")
        img_pil.thumbnail((720, 720), Image.LANCZOS)

        small_img = img_pil.copy()
        small_img.thumbnail((512, 512), Image.BILINEAR)
        buffer = BytesIO()
        small_img.save(buffer, format="JPEG", quality=70)
        img_b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        contents_text = (
            "Rewrite or reply to the highlighted message shown in this image."
            if mode == "rewrite"
            else "Explain the concept, code, or error shown in this image."
            if mode == "explain"
            else "Solve the primary problem shown in this image accurately."
        )
    elif req.text_content:
        contents_text = f"Input:\n{req.text_content}"
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

    # 1. Exact payload dedup (15 min)
    cache_key = hashlib.md5(raw_key.encode()).hexdigest()
    image_hit = IMAGE_CACHE.get(cache_key)
    if not is_followup and image_hit and time.time() - image_hit[0] < CACHE_TTL:
        cached = dict(image_hit[1])
        if not preserve_notes:
            cached["expanded_note"] = ""
        cached["key_source"] = "BYOK" if is_byok else "CACHE"
        return cached

    # 2. Cross-user normalized quiz cache (Memory + 24h SQLite persistence across V1, V2 & V3)
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
            raw_json = route_universal_byok(
                user_keys=user_keys,
                prompt_text=contents_text,
                img_pil=img_pil,
                img_b64=img_b64_str,
                deep=is_followup,
                sys_prompt=sys_prompt,
            )
        except Exception as exc:
            err_msg = str(exc)
            if "BYOK_INVALID" in err_msg:
                return {
                    "detected_type": "ERROR",
                    "cursor_pill": "Invalid key (AIza / gsk_ / nvapi / sk-)",
                    "copy_value": "",
                    "expanded_note": "",
                    "pointer_xy": None,
                    "key_source": "BYOK",
                }
            return {
                "detected_type": "ERROR",
                "cursor_pill": "Personal key busy — retry",
                "copy_value": "",
                "expanded_note": err_msg[:260],
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
        )

    result = extract_json(raw_json, preserve_notes=True)
    confidence = result.pop("confidence", "high")
    result["key_source"] = "BYOK" if is_byok else "POOL"

    IMAGE_CACHE[cache_key] = (time.time(), dict(result))
    if len(IMAGE_CACHE) > 400:
        IMAGE_CACHE.pop(next(iter(IMAGE_CACHE)))

    if not is_followup and quiz_hash and confidence == "high":
        QUIZ_CACHE[quiz_hash] = (time.time(), dict(result))
        if len(QUIZ_CACHE) > 500:
            QUIZ_CACHE.pop(next(iter(QUIZ_CACHE)))
        _db_put(quiz_hash, dict(result))

    if not preserve_notes and result.get("detected_type") in ("MCQ", "DIRECT_VALUE", "CODE_FIX"):
        result["expanded_note"] = ""

    return result


# ==============================================================================
# V1 / V2 LEGACY ROUTE (AUTO-BENEFITS FROM QUIZ DEDUP & 4-PROVIDER CASCADE)
# ==============================================================================
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


# ==============================================================================
# V3 SMART ROUTE (REWRITE / EXPLAIN / SIDECAR NOTES + SAME UNIFIED QUOTA ENGINE)
# ==============================================================================
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

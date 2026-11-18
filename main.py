import os
import re
import json
import base64
import time
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

app = FastAPI(title="Cursor Companion Brain", version="3.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_HTTP_CLIENT = httpx.Client(timeout=20.0)

IMAGE_CACHE = {}
CACHE_TTL = 600
KEY_COOLDOWN = {}
_KEY_CYCLE = itertools.count()

_GEMINI_CLIENTS = {}
_GEMINI_CLIENTS_LOCK = threading.Lock()

GEMINI_FAST_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
]

GEMINI_VISION_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

GROQ_TEXT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]

GROQ_VISION_MODELS = [
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
]

OPENROUTER_TEXT_MODELS = [
    "google/gemma-4-31b-it:free",
    "openrouter/free",
]

OPENROUTER_VISION_MODELS = [
    "google/gemma-4-31b-it:free",
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
    if api_key.startswith("sk-or-"):
        return "openrouter"
    if api_key.startswith("sk-"):
        return "openai"
    return "unknown"


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None
    user_api_key: Optional[str] = None


def _get_client(api_key: str):
    client = _GEMINI_CLIENTS.get(api_key)
    if client is None:
        with _GEMINI_CLIENTS_LOCK:
            client = _GEMINI_CLIENTS.get(api_key)
            if client is None:
                client = genai.Client(
                    api_key=api_key,
                    http_options={"timeout": 19000},
                )
                _GEMINI_CLIENTS[api_key] = client
    return client


@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/healthz", methods=["GET", "HEAD"])
def health():
    now = time.time()
    return {
        "status": "ok",
        "engine": "v3.6-universal-byok",
        "gemini_keys": len(get_keys("GEMINI_API_KEY")),
        "groq_keys": len(get_keys("GROQ_API_KEY")),
        "cached_items": len(IMAGE_CACHE),
        "active_cooldowns": sum(1 for exp in KEY_COOLDOWN.values() if exp > now),
    }


@app.get("/diag")
def diagnose_keys():
    report = {"gemini": [], "groq": "not_configured"}

    for index, api_key in enumerate(get_keys("GEMINI_API_KEY"), 1):
        try:
            client = _get_client(api_key)
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
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

    byok_guard = analyze_context(
        AnalyzeRequest(
            text_content="What is 2+2?",
            user_api_key="AIzaInvalidTestKey1, gsk_InvalidTestKey2, sk-or-InvalidTestKey3",
        )
    )
    results["universal_byok_guard"] = {
        "status": "200 OK" if byok_guard.get("key_source") == "BYOK" else "FAIL",
        "message": byok_guard.get("cursor_pill"),
    }

    results["empty_input_guard"] = analyze_context(AnalyzeRequest()).get("cursor_pill")
    results["elapsed_ms"] = round((time.time() - start_time) * 1000)
    return results


def extract_json(raw_text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw_text or "", flags=re.I).strip()
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        raise ValueError("No JSON object found in model response")

    data = json.loads(match.group(0))
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("Model response was not a JSON object")

    data.setdefault("detected_type", "DIRECT_VALUE")
    data.setdefault("cursor_pill", "Done")
    data.setdefault("copy_value", "")
    data.setdefault("expanded_note", "")
    data.setdefault("pointer_xy", None)

    if data.get("detected_type") in ("MCQ", "DIRECT_VALUE", "CODE_FIX"):
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


def _try_one_gemini(api_key: str, model_name: str, prompt_text: str, img_pil, use_thinking: bool) -> str:
    cfg_kwargs = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
        "max_output_tokens": 950 if use_thinking else 550,
    }
    if use_thinking:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")

    client = _get_client(api_key)
    contents = [PROMPT, prompt_text] + ([img_pil] if img_pil is not None else [])

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        if response and response.text:
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
                return retry_response.text
        raise


def call_gemini_keys(
    keys: list,
    prompt_text: str,
    img_pil: Optional[Image.Image] = None,
    deep: bool = False,
    is_byok: bool = False,
) -> str:
    if not keys:
        raise RuntimeError("No Gemini keys configured")

    now = time.time()
    start_idx = next(_KEY_CYCLE) % len(keys)
    ordered_keys = keys[start_idx:] + keys[:start_idx]
    # Use higher-reasoning vision models when solving images or follow-ups for maximum accuracy
    models = GEMINI_VISION_MODELS if (deep or img_pil is not None) else GEMINI_FAST_MODELS
    last_error = "Gemini keys cooling down"

    for model_name in models:
        use_thinking = (deep or img_pil is not None) and "lite" not in model_name
        for api_key in ordered_keys:
            if KEY_COOLDOWN.get(api_key, 0) > now:
                continue
            model_cd_key = f"{api_key}:{model_name}"
            if KEY_COOLDOWN.get(model_cd_key, 0) > now:
                continue

            try:
                return _try_one_gemini(api_key, model_name, prompt_text, img_pil, use_thinking)
            except Exception as exc:
                error_text = str(exc)
                last_error = f"{model_name}: {error_text[:120]}"
                err_type = _classify_gemini_error(error_text)

                if err_type == "rate_limit":
                    KEY_COOLDOWN[model_cd_key] = time.time() + 45
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
                    {"type": "text", "text": f"{PROMPT}\n\n{prompt_text}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }]
        else:
            messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": prompt_text},
            ]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if provider_tag == "openrouter":
            headers["HTTP-Referer"] = "https://study-buddy-brain.onrender.com"
            headers["X-Title"] = "Cursor Companion"

        try:
            response = _HTTP_CLIENT.post(
                endpoint,
                json={
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 550,
                },
                headers=headers,
            )
        except Exception as exc:
            last_error = f"{provider_tag} {model_id}: {str(exc)[:100]}"
            continue

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        last_error = f"{provider_tag} {model_id}: HTTP {response.status_code}"
        if response.status_code in (401, 403):
            KEY_COOLDOWN[f"{provider_tag}:{api_key}"] = time.time() + 3600
            if is_byok:
                raise RuntimeError("BYOK_INVALID")
            break
        if response.status_code == 429:
            KEY_COOLDOWN[cooldown_key] = time.time() + 35
        elif response.status_code in (400, 404):
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
) -> str:
    gemini_keys = [k for k in user_keys if detect_key_provider(k) == "gemini"]
    groq_keys = [k for k in user_keys if detect_key_provider(k) == "groq"]
    openrouter_keys = [k for k in user_keys if detect_key_provider(k) == "openrouter"]
    openai_keys = [k for k in user_keys if detect_key_provider(k) == "openai"]

    if not (gemini_keys or groq_keys or openrouter_keys or openai_keys):
        raise RuntimeError("BYOK_INVALID")

    last_err = "BYOK_LIMIT"

    if gemini_keys:
        try:
            return call_gemini_keys(gemini_keys, prompt_text, img_pil, deep=deep, is_byok=True)
        except Exception as exc:
            last_err = str(exc)

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
            )
        except Exception as exc:
            last_err = str(exc)

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
            )
        except Exception as exc:
            last_err = str(exc)

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
            )
        except Exception as exc:
            last_err = str(exc)

    raise RuntimeError(last_err)


@app.post("/analyze")
def analyze_context(req: AnalyzeRequest):
    try:
        contents_text = ""
        img_pil = None
        img_b64_str = None
        raw_key = ""
        is_followup = bool(req.follow_up and req.prior_context)
        user_keys = parse_user_keys(req.user_api_key)
        is_byok = bool(user_keys)

        if is_followup:
            contents_text = f"Context:\n{req.prior_context}\n\nFollow-up:\n{req.follow_up}"
            raw_key = contents_text
        elif req.image_base64:
            clean_b64 = req.image_base64.split(",")[-1]
            raw_key = hashlib.md5(clean_b64.encode()).hexdigest()

            img_pil = Image.open(BytesIO(base64.b64decode(clean_b64))).convert("RGB")
            img_pil.thumbnail((720, 720), Image.LANCZOS)

            small_img = img_pil.copy()
            small_img.thumbnail((512, 512), Image.BILINEAR)
            buffer = BytesIO()
            small_img.save(buffer, format="JPEG", quality=70)
            img_b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            contents_text = "Solve the primary problem shown in this image accurately."
        elif req.text_content:
            contents_text = f"Solve:\n{req.text_content}"
            raw_key = req.text_content
        else:
            return {
                "detected_type": "ERROR",
                "cursor_pill": "Select text or snip area first",
                "copy_value": "",
                "expanded_note": "",
                "pointer_xy": None,
                "key_source": "BYOK" if is_byok else "POOL",
            }

        cache_key = hashlib.md5(raw_key.encode()).hexdigest()
        image_hit = IMAGE_CACHE.get(cache_key)
        if not is_followup and image_hit and time.time() - image_hit[0] < CACHE_TTL:
            cached = dict(image_hit[1])
            cached["key_source"] = "BYOK" if is_byok else "POOL"
            return cached

        raw_json = None
        errors = []

        if is_byok:
            try:
                raw_json = route_universal_byok(
                    user_keys=user_keys,
                    prompt_text=contents_text,
                    img_pil=img_pil,
                    img_b64=img_b64_str,
                    deep=is_followup,
                )
            except Exception as exc:
                err_msg = str(exc)
                if "BYOK_INVALID" in err_msg:
                    return {
                        "detected_type": "ERROR",
                        "cursor_pill": "Invalid key (use AIza / gsk_ / sk-)",
                        "copy_value": "",
                        "expanded_note": "",
                        "pointer_xy": None,
                        "key_source": "BYOK",
                    }
                return {
                    "detected_type": "ERROR",
                    "cursor_pill": "Personal key busy — retry",
                    "copy_value": "",
                    "expanded_note": "",
                    "pointer_xy": None,
                    "key_source": "BYOK",
                }
        else:
            try:
                raw_json = call_gemini_keys(
                    keys=get_keys("GEMINI_API_KEY"),
                    prompt_text=contents_text,
                    img_pil=img_pil,
                    deep=is_followup,
                    is_byok=False,
                )
            except Exception as exc:
                errors.append(f"Gemini: {exc}")

            if not raw_json:
                for g_key in get_keys("GROQ_API_KEY"):
                    try:
                        raw_json = call_openai_compat(
                            endpoint="https://api.groq.com/openai/v1/chat/completions",
                            api_key=g_key,
                            models=GROQ_VISION_MODELS if img_b64_str else GROQ_TEXT_MODELS,
                            prompt_text=contents_text,
                            img_b64=img_b64_str,
                            provider_tag="groq",
                            is_byok=False,
                        )
                        if raw_json:
                            break
                    except Exception as exc:
                        errors.append(f"Groq: {exc}")

        if not raw_json:
            raise RuntimeError(" | ".join(errors))

        result = extract_json(raw_json)
        result["key_source"] = "BYOK" if is_byok else "POOL"

        IMAGE_CACHE[cache_key] = (time.time(), result)
        if len(IMAGE_CACHE) > 300:
            IMAGE_CACHE.pop(next(iter(IMAGE_CACHE)))

        return result

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

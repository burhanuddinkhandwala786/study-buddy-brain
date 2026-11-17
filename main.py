import os
import re
import json
import base64
import time
import hashlib
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from io import BytesIO
from typing import Optional

from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import httpx
from google import genai
from google.genai import types


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_HTTP_CLIENT = httpx.Client(timeout=15.0)

IMAGE_CACHE = {}
QUESTION_CACHE = {}
CACHE_TTL = 3600

# "<api_key>" means the whole key is forbidden/banned.
# "<api_key>:<model_name>" means that specific model is rate-limited or unavailable.
KEY_COOLDOWN = {}

_GEMINI_CLIENTS = {}
_GEMINI_CLIENTS_LOCK = threading.Lock()

GEMINI_RACE_TIMEOUT = 12.0


# Fast path is tried first: cheaper / higher-quota models.
GEMINI_FAST_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]

# Deep path tries the strongest model first because follow-ups are less latency-sensitive.
GEMINI_DEEP_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

GROQ_TEXT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

GROQ_VISION_MODELS = [
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
]


PROMPT = """You are an elite academic, competitive-exam, and engineering verification engine.
Solve step-by-step internally before outputting JSON:
1. Read the ENTIRE question carefully (watch for NOT, INCORRECT, EXCEPT, and unit conversions).
2. Independently verify the factual or mathematical answer BEFORE looking at any highlighted/selected radio buttons.
3. Match your verified answer to the exact option letter/number and text.

Return ONLY valid JSON:
{"detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"UI_GUIDE"|"LONG_ANSWER"|"CONCEPT_EXPLAIN","normalized_q":"lowercase core question text without option letters","cursor_pill":"...","copy_value":"...","expanded_note":"...","pointer_xy":null}

Rules:
- MCQ: cursor_pill = exact option letter/number + text (e.g. "B. Sports"). copy_value="", expanded_note="".
- DIRECT_VALUE / CODE_FIX: cursor_pill = exact verified answer/fixed line; copy_value = same raw value; expanded_note="".
- LONG_ANSWER: cursor_pill = short title; copy_value & expanded_note = complete structured breakdown.
- CONCEPT_EXPLAIN: cursor_pill = 1-line summary; expanded_note = 3 high-yield bullet points."""


def get_keys(env_name: str):
    return [
        key.strip()
        for key in os.environ.get(env_name, "").split(",")
        if key.strip()
    ]


def norm_hash(text: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    return hashlib.md5(clean.encode()).hexdigest() if len(clean) >= 12 else ""


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None
    user_api_key: Optional[str] = None


@app.api_route("/", methods=["GET", "HEAD"])
def health():
    now = time.time()

    return {
        "status": "ok",
        "engine": "v2.2-parallel-precision",
        "gemini_keys": len(get_keys("GEMINI_API_KEY")),
        "groq_keys": len(get_keys("GROQ_API_KEY")),
        "cached_questions": len(QUESTION_CACHE),
        "cooling_down": [
            key
            for key, expiry in KEY_COOLDOWN.items()
            if expiry > now
        ],
    }


@app.get("/diag")
def diagnose_keys():
    report = {
        "gemini": [],
        "groq": "not_configured",
    }

    for index, api_key in enumerate(get_keys("GEMINI_API_KEY"), 1):
        try:
            client = _get_client(api_key)

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents="Reply OK",
                config=types.GenerateContentConfig(
                    max_output_tokens=10,
                ),
            )

            report["gemini"].append(
                {
                    "key": f"G{index} (...{api_key[-4:]})",
                    "status": "HEALTHY",
                    "reply": (response.text or "").strip(),
                }
            )

        except Exception as exc:
            report["gemini"].append(
                {
                    "key": f"G{index} (...{api_key[-4:]})",
                    "status": "ERROR",
                    "reason": str(exc)[:120],
                }
            )

    groq_keys = get_keys("GROQ_API_KEY")

    if groq_keys:
        try:
            response = _HTTP_CLIENT.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": GROQ_TEXT_MODELS[0],
                    "messages": [
                        {
                            "role": "user",
                            "content": "Reply OK",
                        }
                    ],
                    "max_tokens": 10,
                },
                headers={
                    "Authorization": f"Bearer {groq_keys[0]}",
                    "Content-Type": "application/json",
                },
            )

            if response.status_code == 200:
                report["groq"] = "HEALTHY"
            else:
                report["groq"] = (
                    f"HTTP {response.status_code}: {response.text[:120]}"
                )

        except Exception as exc:
            report["groq"] = f"ERROR: {str(exc)[:120]}"

    return report


@app.get("/selftest")
def self_test():
    results = {}
    start_time = time.time()

    try:
        raw_groq = call_groq(
            'Return JSON: {"detected_type":"DIRECT_VALUE","normalized_q":"test 1+1","cursor_pill":"2","copy_value":"2","expanded_note":"","pointer_xy":null}'
        )

        parsed_groq = extract_json(raw_groq)

        results["groq_pipeline"] = {
            "status": "200 OK",
            "output": parsed_groq.get("cursor_pill"),
        }

    except Exception as exc:
        results["groq_pipeline"] = {
            "status": "FAIL",
            "error": str(exc),
        }

    try:
        raw_gemini = call_gemini_pool(
            'Return JSON: {"detected_type":"DIRECT_VALUE","normalized_q":"test 2+2","cursor_pill":"4","copy_value":"4","expanded_note":"","pointer_xy":null}'
        )

        parsed_gemini = extract_json(raw_gemini)

        results["gemini_pipeline"] = {
            "status": "200 OK",
            "output": parsed_gemini.get("cursor_pill"),
        }

    except Exception as exc:
        results["gemini_pipeline"] = {
            "status": "FAIL",
            "error": str(exc),
        }

    QUESTION_CACHE[norm_hash("capital of gujarat")] = (
        time.time(),
        {
            "detected_type": "DIRECT_VALUE",
            "cursor_pill": "Gandhinagar",
            "copy_value": "Gandhinagar",
            "expanded_note": "",
            "pointer_xy": None,
        },
    )

    cache_hit = QUESTION_CACHE.get(norm_hash("Capital of Gujarat?!"))

    results["semantic_cache"] = (
        "200 OK (0-token hit)"
        if cache_hit
        else "FAIL"
    )

    results["empty_input_guard"] = analyze_context(
        AnalyzeRequest()
    ).get("cursor_pill")

    results["elapsed_ms"] = round(
        (time.time() - start_time) * 1000
    )

    return results


def extract_json(raw_text: str) -> dict:
    cleaned = re.sub(
        r"```(?:json)?|```",
        "",
        raw_text or "",
        flags=re.I,
    ).strip()

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

    return data


def _get_client(api_key: str):
    """
    Reuse one Gemini client per API key.
    Thread-safe for concurrent FastAPI requests.
    """
    client = _GEMINI_CLIENTS.get(api_key)

    if client is None:
        with _GEMINI_CLIENTS_LOCK:
            client = _GEMINI_CLIENTS.get(api_key)

            if client is None:
                client = genai.Client(
                    api_key=api_key,
                    http_options={"timeout": 9000},
                )
                _GEMINI_CLIENTS[api_key] = client

    return client


def _classify_gemini_error(error_text: str) -> str:
    lowered = error_text.lower()

    if "429" in error_text or "resource_exhausted" in lowered:
        return "rate_limit"

    if "403" in error_text or "permission_denied" in lowered:
        return "forbidden"

    if (
        "404" in error_text
        or "not_found" in lowered
        or "decommission" in lowered
    ):
        return "model_unavailable"

    return "other"


def _try_one(
    api_key: str,
    model_name: str,
    prompt_text: str,
    img_pil,
    use_thinking: bool,
) -> str:
    cfg_kwargs = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
        "max_output_tokens": 1100,
    }

    if use_thinking:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level="low"
        )

    client = _get_client(api_key)

    contents = [PROMPT, prompt_text] + (
        [img_pil] if img_pil is not None else []
    )

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

        # Retry once without thinking if the model rejects the thinking config.
        if use_thinking and (
            "thinking" in error_text.lower()
            or "invalid_argument" in error_text.lower()
        ):
            cfg_kwargs.pop("thinking_config", None)

            retry_response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )

            if retry_response and retry_response.text:
                return retry_response.text

        raise


def call_gemini_pool(
    prompt_text: str,
    img_pil: Optional[Image.Image] = None,
    extra_key: Optional[str] = None,
    deep: bool = False,
) -> str:
    keys = (
        ([extra_key.strip()] if extra_key and extra_key.strip() else [])
        + get_keys("GEMINI_API_KEY")
    )

    if not keys:
        raise RuntimeError("No Gemini keys")

    now = time.time()

    # Remove fully banned/cooled-down keys.
    live_keys = [
        key
        for key in keys
        if KEY_COOLDOWN.get(key, 0) <= now
    ]

    if not live_keys:
        raise RuntimeError("All Gemini keys are cooling down")

    models = GEMINI_DEEP_MODELS if deep else GEMINI_FAST_MODELS
    primary_model = models[0]

    # Do not spend thinking tokens on lite models.
    primary_use_thinking = deep and "lite" not in primary_model

    last_error = "No Gemini attempt succeeded"

    # ---------------------------------------------------------
    # Round 1: race the primary model across all live keys.
    # ---------------------------------------------------------
    with ThreadPoolExecutor(max_workers=len(live_keys)) as executor:
        future_to_key = {
            executor.submit(
                _try_one,
                api_key,
                primary_model,
                prompt_text,
                img_pil,
                primary_use_thinking,
            ): api_key
            for api_key in live_keys
        }

        race_deadline = time.time() + GEMINI_RACE_TIMEOUT

        try:
            for future in as_completed(
                future_to_key,
                timeout=GEMINI_RACE_TIMEOUT,
            ):
                api_key = future_to_key[future]
                remaining_time = max(
                    0.0,
                    race_deadline - time.time(),
                )

                try:
                    return future.result(timeout=remaining_time)

                except Exception as exc:
                    error_text = str(exc)
                    last_error = (
                        f"{primary_model}: {error_text[:160]}"
                    )

                    error_type = _classify_gemini_error(error_text)

                    if error_type == "rate_limit":
                        KEY_COOLDOWN[
                            f"{api_key}:{primary_model}"
                        ] = time.time() + 35

                    elif error_type == "forbidden":
                        KEY_COOLDOWN[api_key] = time.time() + 86400

                    elif error_type == "model_unavailable":
                        KEY_COOLDOWN[
                            f"{api_key}:{primary_model}"
                        ] = time.time() + 3600

        except TimeoutError:
            last_error = (
                f"{primary_model}: no key responded within "
                f"{GEMINI_RACE_TIMEOUT:.0f}s"
            )

        # Cancel unfinished futures so the executor can shut down cleanly.
        for future in future_to_key:
            future.cancel()

    # ---------------------------------------------------------
    # Round 2: sequential fallback through remaining models.
    # ---------------------------------------------------------
    for model_name in models[1:]:
        use_thinking = deep and "lite" not in model_name

        for api_key in live_keys:
            cooldown_key = f"{api_key}:{model_name}"

            if KEY_COOLDOWN.get(cooldown_key, 0) > time.time():
                continue

            try:
                return _try_one(
                    api_key=api_key,
                    model_name=model_name,
                    prompt_text=prompt_text,
                    img_pil=img_pil,
                    use_thinking=use_thinking,
                )

            except Exception as exc:
                error_text = str(exc)
                last_error = f"{model_name}: {error_text[:160]}"

                error_type = _classify_gemini_error(error_text)

                if error_type == "rate_limit":
                    KEY_COOLDOWN[cooldown_key] = time.time() + 35

                elif error_type == "forbidden":
                    KEY_COOLDOWN[api_key] = time.time() + 86400

                elif error_type == "model_unavailable":
                    KEY_COOLDOWN[cooldown_key] = time.time() + 3600

    raise RuntimeError(last_error)


def call_groq(
    prompt_text: str,
    img_b64: Optional[str] = None,
) -> str:
    keys = get_keys("GROQ_API_KEY")

    if not keys:
        raise RuntimeError("No Groq keys")

    models_to_try = (
        GROQ_VISION_MODELS
        if img_b64
        else GROQ_TEXT_MODELS
    )

    last_error = "No Groq attempt succeeded"

    for groq_key in keys:
        for model_id in models_to_try:
            cooldown_key = f"groq:{groq_key}:{model_id}"

            if KEY_COOLDOWN.get(cooldown_key, 0) > time.time():
                continue

            if img_b64:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{PROMPT}\n\n{prompt_text}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt_text,
                    },
                ]

            try:
                response = _HTTP_CLIENT.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 700,
                    },
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json",
                    },
                )

            except Exception as exc:
                last_error = (
                    f"Groq {model_id}: {str(exc)[:120]}"
                )
                continue

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]

            response_text = response.text or ""
            last_error = (
                f"Groq {model_id}: HTTP {response.status_code}: "
                f"{response_text[:120]}"
            )

            if response.status_code == 429:
                KEY_COOLDOWN[cooldown_key] = time.time() + 30

            elif response.status_code in (400, 404) and (
                "decommission" in response_text.lower()
                or "does not exist" in response_text.lower()
                or "model_not_found" in response_text.lower()
            ):
                KEY_COOLDOWN[cooldown_key] = time.time() + 86400

    raise RuntimeError(last_error)


@app.post("/analyze")
def analyze_context(req: AnalyzeRequest):
    try:
        contents_text = ""
        img_pil = None
        img_b64_str = None
        raw_key = ""

        is_followup = bool(
            req.follow_up and req.prior_context
        )

        if is_followup:
            contents_text = (
                f"Context:\n{req.prior_context}\n\n"
                f"Follow-up:\n{req.follow_up}"
            )
            raw_key = contents_text

        elif req.image_base64:
            clean_b64 = req.image_base64.split(",")[-1]
            raw_key = hashlib.md5(clean_b64.encode()).hexdigest()

            img_pil = Image.open(
                BytesIO(base64.b64decode(clean_b64))
            ).convert("RGB")

            img_pil.thumbnail(
                (900, 900),
                Image.LANCZOS,
            )

            small_img = img_pil.copy()
            small_img.thumbnail(
                (480, 480),
                Image.BILINEAR,
            )

            buffer = BytesIO()
            small_img.save(
                buffer,
                format="JPEG",
                quality=65,
            )

            img_b64_str = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

            contents_text = (
                "Solve the problem shown in this image accurately."
            )

        elif req.text_content:
            contents_text = f"Solve:\n{req.text_content}"
            raw_key = req.text_content

            question_hash = norm_hash(req.text_content)

            if (
                question_hash
                and question_hash in QUESTION_CACHE
            ):
                cached_at, cached_result = QUESTION_CACHE[
                    question_hash
                ]

                if time.time() - cached_at < CACHE_TTL:
                    return cached_result

        else:
            return {
                "detected_type": "ERROR",
                "cursor_pill": "Select text or snip area first",
                "copy_value": "",
                "expanded_note": "",
                "pointer_xy": None,
            }

        cache_key = hashlib.md5(raw_key.encode()).hexdigest()

        image_hit = IMAGE_CACHE.get(cache_key)

        if image_hit and time.time() - image_hit[0] < CACHE_TTL:
            return image_hit[1]

        raw_json = None
        errors = []

        # 1. Primary: Gemini pool.
        try:
            raw_json = call_gemini_pool(
                prompt_text=contents_text,
                img_pil=img_pil,
                extra_key=req.user_api_key,
                deep=is_followup,
            )

        except Exception as exc:
            errors.append(f"Gemini: {exc}")

        # 2. Automatic failover: Groq.
        if not raw_json:
            try:
                raw_json = call_groq(
                    contents_text,
                    img_b64_str,
                )

            except Exception as exc:
                errors.append(f"Groq: {exc}")

        if not raw_json:
            raise RuntimeError(" | ".join(errors))

        result = extract_json(raw_json)

        normalized_question = result.pop("normalized_q", "")

        if not is_followup and normalized_question:
            question_hash = norm_hash(normalized_question)

            if question_hash:
                if question_hash in QUESTION_CACHE:
                    cached_at, cached_result = QUESTION_CACHE[
                        question_hash
                    ]

                    if time.time() - cached_at < CACHE_TTL:
                        return cached_result

                QUESTION_CACHE[question_hash] = (
                    time.time(),
                    result,
                )

                if len(QUESTION_CACHE) > 1000:
                    QUESTION_CACHE.pop(
                        next(iter(QUESTION_CACHE))
                    )

        IMAGE_CACHE[cache_key] = (
            time.time(),
            result,
        )

        if len(IMAGE_CACHE) > 500:
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
        }

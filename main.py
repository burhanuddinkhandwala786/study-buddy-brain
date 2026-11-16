import os
import json
import base64
import time
import hashlib
from io import BytesIO
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from google import genai
from google.genai import types

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKING_MODEL = None
_CANDIDATES = {"models": [], "ts": 0}
COOLDOWN = {}
RESULT_CACHE = {}
CACHE_TTL = 600
COOLDOWN_SECS = 65

PREFERRED = [
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]
BLOCKED_WORDS = [
    "embedding", "imagen", "veo", "aqa", "tts", "image",
    "live", "audio", "robotics", "computer-use", "learnlm", "gemma"
]


def get_candidate_models(client):
    if _CANDIDATES["models"] and time.time() - _CANDIDATES["ts"] < 3600:
        return _CANDIDATES["models"]

    available = []
    try:
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            name = (m.name or "").replace("models/", "")
            low = name.lower()
            if (
                "generateContent" in actions
                and low.startswith("gemini")
                and not any(x in low for x in BLOCKED_WORDS)
            ):
                available.append(name)
    except Exception:
        pass

    if available:
        ordered = [m for m in PREFERRED if m in available]
        rest = sorted(
            [m for m in available if m not in ordered],
            key=lambda n: ("lite" not in n, "flash" not in n, n),
        )
        ordered += rest
    else:
        ordered = list(PREFERRED)

    _CANDIDATES["models"] = ordered
    _CANDIDATES["ts"] = time.time()
    return ordered


def is_model_problem(err_str: str) -> bool:
    s = err_str.lower()
    return any(
        k in s
        for k in [
            "404", "not_found", "not found", "no longer available",
            "deprecated", "is not supported", "429", "resource_exhausted",
            "503", "unavailable", "overloaded", "invalid_argument", "timeout", "deadline"
        ]
    )


def call_gemini_with_fallback(client, contents, deep_think: bool = False):
    global WORKING_MODEL
    models_to_try = []

    if deep_think:
        for m in ["gemini-2.5-flash", "gemini-flash-latest"]:
            if m in get_candidate_models(client):
                models_to_try.append(m)
    elif WORKING_MODEL:
        models_to_try.append(WORKING_MODEL)

    for m in get_candidate_models(client):
        if m not in models_to_try:
            models_to_try.append(m)

    last_err = None
    now = time.time()
    ready = [m for m in models_to_try if COOLDOWN.get(m, 0) <= now]
    if not ready:
        raise RuntimeError("Rate limited. Wait 60s and try again.")

    for model_name in ready[:4]:
        try:
            cfg_kwargs = {
                "response_mime_type": "application/json",
                "temperature": 0.0,
                "max_output_tokens": 750 if deep_think else 200,
            }
            if "2.5" in model_name and "lite" not in model_name:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=512 if deep_think else 0
                )

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
            except Exception as inner_e:
                if "thinking_config" in cfg_kwargs:
                    cfg_kwargs.pop("thinking_config", None)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(**cfg_kwargs),
                    )
                else:
                    raise inner_e

            if not deep_think:
                WORKING_MODEL = model_name
            return response.text
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                COOLDOWN[model_name] = time.time() + COOLDOWN_SECS
            if is_model_problem(err_str):
                if model_name == WORKING_MODEL:
                    WORKING_MODEL = None
                continue
            raise

    _CANDIDATES["ts"] = 0
    raise RuntimeError(f"All models failed: {last_err}")


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None


@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "cached_working_model": WORKING_MODEL}


@app.get("/models")
def list_models():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"error": "GEMINI_API_KEY not set"}
    client = genai.Client(api_key=api_key)
    return {"candidates": get_candidate_models(client)}


def error_response(msg: str, detail: str = ""):
    return {
        "detected_type": "ERROR",
        "cursor_pill": msg,
        "copy_value": "",
        "expanded_note": detail,
        "pointer_xy": None,
    }


PROMPT = """Verify the facts and logic carefully before answering. Return valid JSON only:
{"detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"UI_GUIDE"|"LONG_ANSWER"|"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"...","pointer_xy":null}
Rules:
- MCQ: cursor_pill = exact option number/letter + text (e.g. "1. Both A and R are correct..."). copy_value="", expanded_note="".
- DIRECT_VALUE / CODE_FIX: cursor_pill = exact verified answer/fixed line; copy_value = same raw value; expanded_note="".
- LONG_ANSWER (marks >= 3, Differentiate, Compare, Derive, Explain in detail, or follow-up): cursor_pill = short title; copy_value & expanded_note = complete structured multi-point answer covering every requested parameter.
- CONCEPT_EXPLAIN / UI_GUIDE: cursor_pill = 1-line summary; expanded_note = 3 crisp bullets; pointer_xy = [x,y] 0-1000 if UI button visible else null."""


@app.post("/analyze")
def analyze_context(req: AnalyzeRequest):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return error_response("Set GEMINI_API_KEY in Render")

    try:
        client = genai.Client(api_key=api_key)
        contents = [PROMPT]
        deep_think = bool(req.follow_up)

        if req.follow_up and req.prior_context:
            contents.append(f"Context:\n{req.prior_context}\n\nFollow-up:\n{req.follow_up}")
            key = hashlib.md5((req.prior_context + req.follow_up).encode()).hexdigest()
        elif req.image_base64:
            clean_b64 = req.image_base64.split(",")[-1]
            key = hashlib.md5(clean_b64.encode()).hexdigest()
            img = Image.open(BytesIO(base64.b64decode(clean_b64))).convert("RGB")
            img.thumbnail((560, 560))
            contents.append(img)
        elif req.text_content:
            key = hashlib.md5(req.text_content.encode()).hexdigest()
            low_t = req.text_content.lower()
            if any(w in low_t for w in ["differentiate", "derive", "explain", "compare", "marks", "circuit", "calculate"]):
                deep_think = True
            contents.append(f"Text:\n{req.text_content}")
        else:
            return error_response("Empty input")

        hit = RESULT_CACHE.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]

        raw_json = call_gemini_with_fallback(client, contents, deep_think=deep_think)
        cleaned = (
            raw_json.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(cleaned)
        if isinstance(data, list) and data:
            data = data[0]

        RESULT_CACHE[key] = (time.time(), data)
        if len(RESULT_CACHE) > 200:
            RESULT_CACHE.pop(next(iter(RESULT_CACHE)))

        return data
    except Exception as e:
        return error_response(f"Err: {str(e)[:55]}", str(e))

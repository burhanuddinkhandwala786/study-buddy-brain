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
COOLDOWN = {}      # model name -> unix time until which we skip it (after a 429)
RESULT_CACHE = {}  # input hash -> (timestamp, result), so repeat captures cost nothing
CACHE_TTL = 600
COOLDOWN_SECS = 65

# "-latest" aliases are maintained by Google and don't get retired like pinned names.
PREFERRED = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
BLOCKED_WORDS = ["embedding", "imagen", "veo", "aqa", "tts", "image", "live",
                 "audio", "robotics", "computer-use", "learnlm", "gemma"]


def get_candidate_models(client):
    # Cache the model list for 1 hour so we don't hit models.list() on every request
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
        # Only keep preferred models that this API key can actually see,
        # then fall back to anything else that's available (flash first, newest first).
        ordered = [m for m in PREFERRED if m in available]
        rest = sorted(
            [m for m in available if m not in ordered],
            key=lambda n: ("flash" not in n, n),
        )
        ordered += rest
    else:
        # models.list() failed, so just try the preferred names blindly
        ordered = list(PREFERRED)

    _CANDIDATES["models"] = ordered
    _CANDIDATES["ts"] = time.time()
    return ordered


def is_model_problem(err_str: str) -> bool:
    s = err_str.lower()
    return any(k in s for k in ["404", "not_found", "not found", "no longer available",
                                "deprecated", "is not supported", "429", "resource_exhausted",
                                "503", "unavailable", "overloaded"])


def call_gemini_with_fallback(client, contents):
    global WORKING_MODEL
    models_to_try = [WORKING_MODEL] if WORKING_MODEL else []
    for m in get_candidate_models(client):
        if m not in models_to_try:
            models_to_try.append(m)

    last_err = None
    now = time.time()
    ready = [m for m in models_to_try if COOLDOWN.get(m, 0) <= now]
    if not ready:
        raise RuntimeError("Rate limited. Wait a minute and try again")

    for model_name in ready[:8]:  # cap attempts so a request can't hang forever
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
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

    # Everything failed: force a fresh model list next time
    _CANDIDATES["ts"] = 0
    raise RuntimeError(f"All models failed. Last error: {last_err}")


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None


@app.get("/")
def health():
    return {"status": "ok", "cached_working_model": WORKING_MODEL}


@app.get("/models")
def list_models():
    """Debug endpoint: shows which models your API key can actually use."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"error": "GEMINI_API_KEY not set"}
    client = genai.Client(api_key=api_key)
    return {"candidates": get_candidate_models(client)}


def error_response(msg, detail=""):
    return {
        "detected_type": "ERROR",
        "cursor_pill": msg,
        "copy_value": "",
        "expanded_note": detail,
        "pointer_xy": None,
    }


PROMPT = """
You are an ultra-fast, context-aware cursor intelligence. Analyze the input strictly by context and return valid JSON with keys:
- "detected_type": "MCQ" | "DIRECT_VALUE" | "CODE_FIX" | "UI_GUIDE" | "CONCEPT_EXPLAIN"
- "cursor_pill": Ultra-short answer displayed directly on the user's cursor.
  * If MCQ: ONLY the exact option letter and text (e.g., "B) Half-wave dipole"). Zero explanation.
  * If Numerical / Fill-in-blank: ONLY the exact final value/unit.
  * If Code error: ONLY the corrected line of code.
  * If UI/Software screen asking how/where to do something: 1-line instruction (e.g., "Click Export -> PDF").
  * If Open-ended theory/diagram: 1 crisp sentence core takeaway.
- "copy_value": The clean raw value/code/answer to auto-copy to clipboard (empty string for MCQs).
- "expanded_note": If CONCEPT_EXPLAIN or UI_GUIDE, give a concise 3-bullet step-by-step guide; otherwise "".
- "pointer_xy": If UI_GUIDE and a specific button/icon is visible in the image, return [x, y] normalized from 0 to 1000; otherwise null.
If the screen contains several questions, answer the one closest to the cursor, or the first one if unclear.
"""


@app.post("/analyze")
def analyze_context(req: AnalyzeRequest):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return error_response("Set GEMINI_API_KEY in Render")

    try:
        client = genai.Client(api_key=api_key)

        contents = [PROMPT]
        if req.image_base64:
            clean_b64 = req.image_base64.split(",")[-1]
            key = hashlib.md5(clean_b64.encode()).hexdigest()
            img = Image.open(BytesIO(base64.b64decode(clean_b64))).convert("RGB")
            img.thumbnail((1280, 1280))  # smaller image = fewer tokens = less quota used
            contents.append(img)
        elif req.text_content:
            key = hashlib.md5(req.text_content.encode()).hexdigest()
            contents.append(f"Selected text:\n{req.text_content}")
        else:
            return error_response("Empty input")

        hit = RESULT_CACHE.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]

        raw_json = call_gemini_with_fallback(client, contents)

        # Strip code fences if the model adds them, and handle a list-wrapped reply
        cleaned = raw_json.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        if isinstance(data, list) and data:
            data = data[0]
        RESULT_CACHE[key] = (time.time(), data)
        if len(RESULT_CACHE) > 200:
            RESULT_CACHE.pop(next(iter(RESULT_CACHE)))
        return data
    except Exception as e:
        return error_response(f"Err: {str(e)[:55]}", str(e))

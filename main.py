import os
import re
import json
import base64
import time
import hashlib
import itertools
from io import BytesIO
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
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

_HTTP_CLIENT = httpx.Client(timeout=20.0)
RESULT_CACHE = {}
CACHE_TTL = 900
KEY_COOLDOWN = {}

# High-quota free-tier Gemini models FIRST (never put 0-quota 3.x models ahead of 2.5)
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

PROMPT = """You are an expert academic and technical verification engine.
Verify formulas, Indian/regional GK awards, option letters, and logical constraints carefully—never be misled by already-clicked radio buttons. Return ONLY valid JSON:
{"detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"UI_GUIDE"|"LONG_ANSWER"|"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"...","pointer_xy":null}
Rules:
- MCQ: cursor_pill = exact option letter/number + text (e.g. "4. Bharuch"). copy_value="", expanded_note="".
- DIRECT_VALUE / CODE_FIX: cursor_pill = exact verified answer/fixed line; copy_value = same raw value; expanded_note="".
- LONG_ANSWER: cursor_pill = short title; copy_value & expanded_note = complete structured breakdown.
- CONCEPT_EXPLAIN: cursor_pill = 1-line summary; expanded_note = 3 high-yield bullet points."""


def get_gemini_keys():
    raw = os.environ.get("GEMINI_API_KEY", "")
    # Supports either 1 key or multiple comma-separated keys (key1,key2,key3) for 3x-5x free quota
    return [k.strip() for k in raw.split(",") if k.strip()]


_KEY_CYCLE = itertools.count()


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None


@app.api_route("/", methods=["GET", "HEAD"])
def health():
    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return {
        "status": "ok",
        "gemini_keys_loaded": len(get_gemini_keys()),
        "groq_ready": bool(os.environ.get("GROQ_API_KEY", "").strip()),
        "openrouter_ready": or_key.startswith("sk-or-"),
    }


def extract_json(raw_text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.S).strip()
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        raise ValueError("Invalid JSON")
    data = json.loads(match.group(0))
    if isinstance(data, list) and data:
        data = data[0]
    data.setdefault("detected_type", "DIRECT_VALUE")
    data.setdefault("cursor_pill", "Done")
    data.setdefault("copy_value", "")
    data.setdefault("expanded_note", "")
    data.setdefault("pointer_xy", None)
    return data


def call_gemini_pool(prompt_text: str, img_pil: Optional[Image.Image] = None) -> str:
    keys = get_gemini_keys()
    if not keys:
        raise RuntimeError("No GEMINI_API_KEY configured")

    now = time.time()
    start_idx = next(_KEY_CYCLE) % len(keys)
    ordered_keys = keys[start_idx:] + keys[:start_idx]
    last_err = "No active Gemini quota"

    for idx, api_key in enumerate(ordered_keys):
        if KEY_COOLDOWN.get(api_key, 0) > now:
            continue
        client = genai.Client(api_key=api_key, http_options={"timeout": 18000})
        for model_name in GEMINI_MODELS:
            contents = [PROMPT, prompt_text]
            if img_pil is not None:
                contents.append(img_pil)
            try:
                cfg_kwargs = {
                    "response_mime_type": "application/json",
                    "temperature": 0.0,
                    "max_output_tokens": 1000,
                }
                if model_name == "gemini-2.5-flash":
                    cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=256)

                res = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
                if res and res.text:
                    return res.text
            except Exception as e:
                err_s = str(e)
                last_err = f"G-key{idx+1}/{model_name}: {err_s[:55]}"
                if "429" in err_s or "RESOURCE_EXHAUSTED" in err_s:
                    KEY_COOLDOWN[api_key] = time.time() + 45
                    break  # Rotate immediately to next Gemini key
                if "403" in err_s or "PERMISSION_DENIED" in err_s:
                    KEY_COOLDOWN[api_key] = time.time() + 3600
                    break
                continue

    raise RuntimeError(last_err)


def call_groq(prompt_text: str, image_b64: Optional[str] = None) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("No GROQ_API_KEY")

    if image_b64:
        model_id = "qwen/qwen3.8-27b"
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{PROMPT}\n\n{prompt_text}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }]
    else:
        model_id = "llama-3.3-70b-versatile"
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": prompt_text},
        ]

    for attempt in range(2):
        res = _HTTP_CLIENT.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={"model": model_id, "messages": messages, "temperature": 0.0, "max_tokens": 600},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        if res.status_code == 429 and attempt == 0:
            time.sleep(1.5)
            continue
        raise RuntimeError(f"Groq {res.status_code}")


def call_openrouter_free(prompt_text: str, image_b64: Optional[str] = None) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or not key.startswith("sk-or-"):
        raise RuntimeError("No valid OPENROUTER_API_KEY")

    model_id = "qwen/qwen2.5-vl-72b-instruct:free" if image_b64 else "meta-llama/llama-3.3-70b-instruct:free"
    if image_b64:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{PROMPT}\n\n{prompt_text}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }]
    else:
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": prompt_text},
        ]

    res = _HTTP_CLIENT.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json={"model": model_id, "messages": messages, "temperature": 0.0, "max_tokens": 700},
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://cursor-companion.app",
            "X-Title": "Cursor Companion",
        },
    )
    if res.status_code != 200:
        raise RuntimeError(f"OpenRouter {res.status_code}")
    return res.json()["choices"][0]["message"]["content"]


@app.post("/analyze")
def analyze_context(req: AnalyzeRequest):
    try:
        contents_text = ""
        img_pil = None
        img_b64_str = None
        raw_key = ""

        if req.follow_up and req.prior_context:
            contents_text = f"Context:\n{req.prior_context}\n\nFollow-up:\n{req.follow_up}"
            raw_key = contents_text
        elif req.image_base64:
            clean_b64 = req.image_base64.split(",")[-1]
            raw_key = hashlib.md5(clean_b64.encode()).hexdigest()
            img_pil = Image.open(BytesIO(base64.b64decode(clean_b64))).convert("RGB")
            img_pil.thumbnail((600, 600), Image.BILINEAR)

            groq_img = img_pil.copy()
            groq_img.thumbnail((420, 420), Image.BILINEAR)
            buf = BytesIO()
            groq_img.save(buf, format="JPEG", quality=62)
            img_b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

            contents_text = "Solve the problem shown in this image accurately."
        elif req.text_content:
            contents_text = f"Solve:\n{req.text_content}"
            raw_key = req.text_content
        else:
            return {"detected_type": "ERROR", "cursor_pill": "Select text or snip area first", "copy_value": "", "expanded_note": "", "pointer_xy": None}

        key = hashlib.md5(raw_key.encode()).hexdigest()
        hit = RESULT_CACHE.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]

        raw_json = None
        errors = []

        # 1. Gemini Key Pool (gemini-2.5-flash-lite -> gemini-2.5-flash)
        try:
            raw_json = call_gemini_pool(contents_text, img_pil)
        except Exception as e:
            errors.append(str(e))

        # 2. Groq Backup
        if not raw_json:
            try:
                raw_json = call_groq(contents_text, img_b64_str)
            except Exception as e:
                errors.append(str(e))

        # 3. OpenRouter Free Backup
        if not raw_json:
            try:
                raw_json = call_openrouter_free(contents_text, img_b64_str)
            except Exception as e:
                errors.append(str(e))

        if not raw_json:
            raise RuntimeError(" | ".join(errors))

        data = extract_json(raw_json)
        RESULT_CACHE[key] = (time.time(), data)
        if len(RESULT_CACHE) > 300:
            RESULT_CACHE.pop(next(iter(RESULT_CACHE)))

        return data
    except Exception as e:
        print(f"[BACKEND ERR]: {e}")
        return {
            "detected_type": "ERROR",
            "cursor_pill": "Server busy — press hotkey again",
            "copy_value": "",
            "expanded_note": str(e),
            "pointer_xy": None,
        }

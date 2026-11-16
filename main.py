import os
import re
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

# Reuse single HTTP/GenAI clients across requests so TLS handshakes don't repeat
_GEMINI_CLIENT = None
_HTTP_CLIENT = httpx.Client(timeout=20.0)
RESULT_CACHE = {}
CACHE_TTL = 600

PROMPT = """You are an expert academic and technical verification engine.
Verify formulas, option numbers, and logical constraints carefully before answering. Return ONLY valid JSON:
{"detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"UI_GUIDE"|"LONG_ANSWER"|"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"...","pointer_xy":null}
Rules:
- MCQ: cursor_pill = exact option letter/number + text (e.g. "2. Stratified"). copy_value="", expanded_note="".
- DIRECT_VALUE / CODE_FIX: cursor_pill = exact verified answer/fixed line; copy_value = same raw value; expanded_note="".
- LONG_ANSWER: cursor_pill = short title; copy_value & expanded_note = complete structured breakdown.
- CONCEPT_EXPLAIN: cursor_pill = 1-line summary; expanded_note = 3 high-yield bullet points."""


def get_gemini():
    global _GEMINI_CLIENT
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client(api_key=key)
    return _GEMINI_CLIENT


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None


@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "system": "fast-direct-v2.5"}


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


def call_groq_fast(prompt_text: str, image_b64: Optional[str] = None) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("No GROQ_API_KEY")

    if image_b64:
        model_id = "meta-llama/llama-4-scout-17b-16e-instruct"
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

    res = _HTTP_CLIENT.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json={"model": model_id, "messages": messages, "temperature": 0.0, "max_tokens": 1200},
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    if res.status_code != 200:
        raise RuntimeError(f"Groq {res.status_code}")
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
            img_pil.thumbnail((720, 720), Image.BILINEAR)
            buf = BytesIO()
            img_pil.save(buf, format="JPEG", quality=78)
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
        client = get_gemini()

        # Single direct hop to Gemini 2.5 Flash (zero model-listing latency)
        if client:
            for model_name in ("gemini-2.5-flash", "gemini-flash-latest"):
                try:
                    contents = [PROMPT, contents_text]
                    if img_pil is not None:
                        contents.append(img_pil)
                    cfg = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        max_output_tokens=1500,
                        thinking_config=types.ThinkingConfig(thinking_budget=512),
                    )
                    res = client.models.generate_content(model=model_name, contents=contents, config=cfg)
                    if res and res.text:
                        raw_json = res.text
                        break
                except Exception as e:
                    print(f"[{model_name} err]: {e}")
                    continue

        # Instant Groq fallback only if Gemini hits 429/outage
        if not raw_json:
            raw_json = call_groq_fast(contents_text, img_b64_str)

        data = extract_json(raw_json)
        RESULT_CACHE[key] = (time.time(), data)
        if len(RESULT_CACHE) > 200:
            RESULT_CACHE.pop(next(iter(RESULT_CACHE)))

        return data
    except Exception as e:
        print(f"[BACKEND ERR]: {e}")
        return {
            "detected_type": "ERROR",
            "cursor_pill": "High traffic — press hotkey again",
            "copy_value": "",
            "expanded_note": "",
            "pointer_xy": None,
        }

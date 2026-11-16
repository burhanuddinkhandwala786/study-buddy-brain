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

RESULT_CACHE = {}
CACHE_TTL = 600

PROMPT = """You are an expert academic and technical verification engine.
Verify formulas, option numbers, and logical constraints step-by-step. Return ONLY valid JSON with no markdown wrapping or extra text:
{"detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"UI_GUIDE"|"LONG_ANSWER"|"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"...","pointer_xy":null}
Rules:
- MCQ: cursor_pill = exact option letter/number + text (e.g. "2. Stratified"). copy_value="", expanded_note="".
- DIRECT_VALUE / CODE_FIX: cursor_pill = exact verified answer/fixed line; copy_value = same raw value; expanded_note="".
- LONG_ANSWER: cursor_pill = short title; copy_value & expanded_note = rigorous multi-point breakdown.
- CONCEPT_EXPLAIN: cursor_pill = 1-line summary; expanded_note = 3 high-yield bullet points."""


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None


@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {
        "status": "ok",
        "system": "triple-redundancy-enterprise-live",
        "providers": {
            "gemini": bool(os.environ.get("GEMINI_API_KEY")),
            "groq": bool(os.environ.get("GROQ_API_KEY")),
            "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
        },
    }


def extract_json_safely(raw_text: str) -> dict:
    if not raw_text:
        raise ValueError("Empty model response")
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.S).strip()
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        raise ValueError("No valid JSON object found")
    data = json.loads(match.group(0))
    if isinstance(data, list) and data:
        data = data[0]
    data.setdefault("detected_type", "DIRECT_VALUE")
    data.setdefault("cursor_pill", "Solution ready")
    data.setdefault("copy_value", "")
    data.setdefault("expanded_note", "")
    data.setdefault("pointer_xy", None)
    return data


def call_groq(prompt_text: str, image_b64: Optional[str] = None) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY")

    if image_b64:
        model_id = "meta-llama/llama-4-scout-17b-16e-instruct"
        user_msg = [
            {"type": "text", "text": f"{PROMPT}\n\n{prompt_text}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
        messages = [{"role": "user", "content": user_msg}]
    else:
        model_id = "llama-3.3-70b-versatile"
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": prompt_text},
        ]

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    if not image_b64:
        payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=30.0) as client:
        res = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        if res.status_code != 200:
            raise RuntimeError(f"Groq HTTP {res.status_code}: {res.text[:150]}")
        return res.json()["choices"][0]["message"]["content"]


def call_openrouter(prompt_text: str, image_b64: Optional[str] = None) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    if image_b64:
        user_msg = [
            {"type": "text", "text": f"{PROMPT}\n\n{prompt_text}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
        messages = [{"role": "user", "content": user_msg}]
    else:
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": prompt_text},
        ]

    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    with httpx.Client(timeout=30.0) as client:
        res = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://cursor-companion.app",
                "X-Title": "Cursor Companion",
            },
        )
        if res.status_code != 200:
            raise RuntimeError(f"OpenRouter HTTP {res.status_code}: {res.text[:150]}")
        return res.json()["choices"][0]["message"]["content"]


@app.post("/analyze")
def analyze_context(req: AnalyzeRequest):
    try:
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
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
            img_bytes = base64.b64decode(clean_b64)
            img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
            img_pil.thumbnail((840, 840), Image.LANCZOS)
            buf = BytesIO()
            img_pil.save(buf, format="JPEG", quality=85)
            img_b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            contents_text = "Analyze this image and provide the requested JSON output."
        elif req.text_content:
            contents_text = f"Text:\n{req.text_content}"
            raw_key = req.text_content
        else:
            return {
                "detected_type": "ERROR",
                "cursor_pill": "Empty input received",
                "copy_value": "",
                "expanded_note": "Please provide text or image content.",
                "pointer_xy": None,
            }

        key = hashlib.md5(raw_key.encode()).hexdigest()
        hit = RESULT_CACHE.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]

        raw_json = None
        last_error = None

        # --- TIER 1: Gemini 2.5 Flash ---
        if gemini_key:
            try:
                client = genai.Client(api_key=gemini_key)
                contents = [PROMPT, contents_text]
                if img_pil is not None:
                    contents.append(img_pil)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        max_output_tokens=2048,
                    ),
                )
                if response and response.text:
                    raw_json = response.text
            except Exception as e:
                last_error = f"Gemini: {e}"

        # --- TIER 2: Groq Vision / Llama 3.3 70B ---
        if not raw_json:
            try:
                raw_json = call_groq(contents_text, img_b64_str)
            except Exception as e:
                last_error = f"{last_error} | Groq: {e}"

        # --- TIER 3: OpenRouter Failover ---
        if not raw_json:
            try:
                raw_json = call_openrouter(contents_text, img_b64_str)
            except Exception as e:
                raise RuntimeError(f"All 3 AI tiers failed: {last_error} | OpenRouter: {e}")

        data = extract_json_safely(raw_json)

        RESULT_CACHE[key] = (time.time(), data)
        if len(RESULT_CACHE) > 200:
            RESULT_CACHE.pop(next(iter(RESULT_CACHE)))

        return data
    except Exception as e:
        return {
            "detected_type": "ERROR",
            "cursor_pill": f"Err: {str(e)[:45]}",
            "copy_value": "",
            "expanded_note": str(e),
            "pointer_xy": None,
        }

import os
import re
import json
import base64
import time
import hashlib
from io import BytesIO
from typing import Optional, Literal

import httpx
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_IMG_SIDE = 1280
RESULT_CACHE = {}
CACHE_TTL = 600

SYSTEM = """You are an expert academic tutor and engineering assistant. Solve the problem accurately.
You must return valid JSON only, with exactly these keys:
{
  "transcription": "Exact transcription of the problem text/equations from the image or input",
  "reasoning": "Step-by-step logical derivation or verification",
  "detected_type": "MCQ" or "DIRECT_VALUE" or "CODE_FIX" or "UI_GUIDE" or "LONG_ANSWER" or "CONCEPT_EXPLAIN",
  "final_answer": "The concise answer shown on the cursor pill (e.g. '2. Stratified' for MCQ or the exact result)",
  "copy_value": "Value to copy to clipboard (if applicable)",
  "expanded_note": "Detailed breakdown, notes, or steps for long answers",
  "confidence": integer between 0 and 100
}"""

class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None
    mode: Literal["fast", "accurate"] = "accurate"

def parse_json(txt):
    if not txt:
        raise ValueError("Empty response from model")
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```json\s*", "", txt.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    
    m = re.search(r"\{.*\}", cleaned, re.S)
    if not m:
        raise ValueError(f"No JSON object found in response: {txt[:200]}")
    
    d = json.loads(m.group(0))
    d.setdefault("transcription", "")
    d.setdefault("reasoning", "")
    d.setdefault("detected_type", "DIRECT_VALUE")
    d.setdefault("final_answer", "Solution ready")
    d.setdefault("copy_value", "")
    d.setdefault("expanded_note", "")
    try:
        d["confidence"] = int(d.get("confidence", 85))
    except Exception:
        d["confidence"] = 85
    return d

def call_gemini(user_text, img=None):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    
    client = genai.Client(api_key=key)
    contents = [SYSTEM, user_text]
    if img is not None:
        contents.append(img)
        
    models_to_try = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]
    last_err = None
    
    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=4096
                )
            )
            return parse_json(response.text)
        except Exception as e:
            last_err = e
            continue
            
    raise RuntimeError(f"Gemini models failed: {str(last_err)}")

def call_groq(user_text):
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY")
    
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 4096
    }
    
    with httpx.Client(timeout=30.0) as client:
        res = client.post(
            "[https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)",
            headers={"Authorization": f"Bearer {key}"},
            json=body
        )
        if res.status_code != 200:
            raise RuntimeError(f"Groq error {res.status_code}: {res.text[:150]}")
        return parse_json(res.json()["choices"][0]["message"]["content"])

@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {
        "status": "ok",
        "gemini": bool(os.environ.get("GEMINI_API_KEY")),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
    }

def error_response(msg, detail=""):
    return {
        "detected_type": "ERROR",
        "cursor_pill": msg,
        "copy_value": "",
        "expanded_note": detail,
        "pointer_xy": None,
        "confidence": 0,
        "status": "error"
    }

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return error_response("Set GEMINI_API_KEY in Render environment")
        
    try:
        img = None
        if req.follow_up and req.prior_context:
            text_in = f"Earlier context:\n{req.prior_context}\n\nFollow-up question:\n{req.follow_up}"
            raw_key = req.prior_context + req.follow_up
        elif req.image_base64:
            raw_b64 = req.image_base64.split(",")[-1]
            raw = base64.b64decode(raw_b64)
            img = Image.open(BytesIO(raw)).convert("RGB")
            if max(img.size) > MAX_IMG_SIDE:
                img.thumbnail((MAX_IMG_SIDE, MAX_IMG_SIDE), Image.LANCZOS)
            text_in = "Analyze and solve the problem shown in this image accurately."
            raw_key = raw_b64[:100]
        elif req.text_content:
            text_in = f"Solve this:\n{req.text_content}"
            raw_key = req.text_content
        else:
            return error_response("Empty input received")

        key = hashlib.md5((req.mode + raw_key).encode()).hexdigest()
        hit = RESULT_CACHE.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]

        sol = None
        status = "single"
        
        # Try Gemini first
        try:
            sol = call_gemini(text_in, img)
        except Exception as gemini_err:
            # Fallback to Groq if Gemini fails
            try:
                sol = call_groq(text_in)
                status = "fallback-groq"
            except Exception as groq_err:
                return error_response(f"Err: Gemini & Groq failed", f"Gemini: {str(gemini_err)} | Groq: {str(groq_err)}")

        conf = sol.get("confidence", 85)
        pill = sol.get("final_answer", "Done")
        if conf >= 90:
            pill = "✓ " + pill
        elif conf < 60:
            pill = "⚠ " + pill

        data = {
            "detected_type": sol.get("detected_type", "DIRECT_VALUE"),
            "cursor_pill": pill,
            "copy_value": sol.get("copy_value", ""),
            "expanded_note": sol.get("expanded_note", sol.get("reasoning", "")),
            "pointer_xy": None,
            "confidence": conf,
            "status": status,
        }

        RESULT_CACHE[key] = (time.time(), data)
        if len(RESULT_CACHE) > 200:
            RESULT_CACHE.pop(next(iter(RESULT_CACHE)))
        return data

    except Exception as e:
        return error_response(f"Err: {str(e)[:55]}", str(e))

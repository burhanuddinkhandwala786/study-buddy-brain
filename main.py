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

_GEMINI_CLIENT = None
_ACTIVE_GEMINI_MODELS = []
_HTTP_CLIENT = httpx.Client(timeout=20.0)
RESULT_CACHE = {}
CACHE_TTL = 600

PROMPT = """You are an expert academic and technical verification engine.
Verify formulas, Indian/regional GK awards, option letters, and logical constraints carefully—never be misled by already-clicked radio buttons. Return ONLY valid JSON:
{"detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"UI_GUIDE"|"LONG_ANSWER"|"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"...","pointer_xy":null}
Rules:
- MCQ: cursor_pill = exact option letter/number + text (e.g. "4. Bharuch"). copy_value="", expanded_note="".
- DIRECT_VALUE / CODE_FIX: cursor_pill = exact verified answer/fixed line; copy_value = same raw value; expanded_note="".
- LONG_ANSWER: cursor_pill = short title; copy_value & expanded_note = complete structured breakdown.
- CONCEPT_EXPLAIN: cursor_pill = 1-line summary; expanded_note = 3 high-yield bullet points."""


def init_providers():
    global _GEMINI_CLIENT, _ACTIVE_GEMINI_MODELS
    g_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if g_key and _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client(api_key=g_key, http_options={"timeout": 20000})
        try:
            found = []
            for m in _GEMINI_CLIENT.models.list():
                name = (m.name or "").replace("models/", "")
                actions = getattr(m, "supported_actions", None) or []
                low = name.lower()
                if "generateContent" in actions and low.startswith("gemini"):
                    if not any(b in low for b in ["image", "audio", "tts", "live", "embedding", "gemma", "robotics", "computer-use", "omni", "preview"]):
                        found.append(name)
            flash = sorted([n for n in found if "flash" in n and "lite" not in n], reverse=True)
            lite = sorted([n for n in found if "flash" in n and "lite" in n], reverse=True)
            _ACTIVE_GEMINI_MODELS = flash + lite
        except Exception:
            pass
        if not _ACTIVE_GEMINI_MODELS:
            _ACTIVE_GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]


@app.on_event("startup")
def on_startup():
    init_providers()


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None


@app.api_route("/", methods=["GET", "HEAD"])
def health():
    if not _ACTIVE_GEMINI_MODELS:
        init_providers()
    return {
        "status": "ok",
        "gemini_models": _ACTIVE_GEMINI_MODELS[:3],
        "groq_ready": bool(os.environ.get("GROQ_API_KEY")),
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

    for attempt in range(3):
        res = _HTTP_CLIENT.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={"model": model_id, "messages": messages, "temperature": 0.0, "max_tokens": 600},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        if res.status_code == 429 and attempt < 2:
            time.sleep(1.5 * (attempt + 1))
            continue
        raise RuntimeError(f"Groq {res.status_code}: {res.text[:100]}")


@app.post("/analyze")
def analyze_context(req: AnalyzeRequest):
    if not _ACTIVE_GEMINI_MODELS:
        init_providers()

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
            img_pil.thumbnail((640, 640), Image.BILINEAR)

            groq_img = img_pil.copy()
            groq_img.thumbnail((448, 448), Image.BILINEAR)
            buf = BytesIO()
            groq_img.save(buf, format="JPEG", quality=65)
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
        last_err = ""

        if _GEMINI_CLIENT:
            for model_name in _ACTIVE_GEMINI_MODELS[:2]:
                contents = [PROMPT, contents_text]
                if img_pil is not None:
                    contents.append(img_pil)
                try:
                    cfg = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        max_output_tokens=1200,
                        thinking_config=types.ThinkingConfig(thinking_budget=256),
                    )
                    res = _GEMINI_CLIENT.models.generate_content(model=model_name, contents=contents, config=cfg)
                    if res and res.text:
                        raw_json = res.text
                        break
                except Exception as e:
                    if "thinking" in str(e).lower():
                        try:
                            cfg_plain = types.GenerateContentConfig(
                                response_mime_type="application/json",
                                temperature=0.0,
                                max_output_tokens=1200,
                            )
                            res = _GEMINI_CLIENT.models.generate_content(model=model_name, contents=contents, config=cfg_plain)
                            if res and res.text:
                                raw_json = res.text
                                break
                        except Exception as e2:
                            last_err = f"{model_name}: {str(e2)[:60]}"
                    else:
                        last_err = f"{model_name}: {str(e)[:60]}"
                    if "403" in last_err or "PERMISSION_DENIED" in last_err or "free tier" in last_err.lower():
                        break
                    continue

        if not raw_json:
            raw_json = call_groq(contents_text, img_b64_str)

        data = extract_json(raw_json)
        RESULT_CACHE[key] = (time.time(), data)
        if len(RESULT_CACHE) > 200:
            RESULT_CACHE.pop(next(iter(RESULT_CACHE)))

        return data
    except Exception as e:
        print(f"[BACKEND ERR]: {e}")
        return {
            "detected_type": "ERROR",
            "cursor_pill": f"Err: {str(e)[:48]}",
            "copy_value": "",
            "expanded_note": str(e),
            "pointer_xy": None,
        }

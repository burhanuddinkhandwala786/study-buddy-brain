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
_ACTIVE_GROQ_VISION = []
_HTTP_CLIENT = httpx.Client(timeout=20.0)
RESULT_CACHE = {}
CACHE_TTL = 600

PROMPT = """You are an expert academic and technical verification engine.
Verify formulas, option numbers, and logical constraints carefully before answering. Return ONLY valid JSON:
{"detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"UI_GUIDE"|"LONG_ANSWER"|"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"...","pointer_xy":null}
Rules:
- MCQ: cursor_pill = exact option letter/number + text (e.g. "4. Bharuch"). copy_value="", expanded_note="".
- DIRECT_VALUE / CODE_FIX: cursor_pill = exact verified answer/fixed line; copy_value = same raw value; expanded_note="".
- LONG_ANSWER: cursor_pill = short title; copy_value & expanded_note = complete structured breakdown.
- CONCEPT_EXPLAIN: cursor_pill = 1-line summary; expanded_note = 3 high-yield bullet points."""


def init_providers():
    global _GEMINI_CLIENT, _ACTIVE_GEMINI_MODELS, _ACTIVE_GROQ_VISION
    g_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if g_key and _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client(api_key=g_key)
        try:
            found = []
            for m in _GEMINI_CLIENT.models.list():
                name = (m.name or "").replace("models/", "")
                actions = getattr(m, "supported_actions", None) or []
                low = name.lower()
                if "generateContent" in actions and low.startswith("gemini"):
                    if not any(b in low for b in ["image", "audio", "tts", "live", "embedding", "gemma", "robotics", "computer-use"]):
                        found.append(name)
            # Prefer newest flash models available on THIS specific API key
            flash = sorted([n for n in found if "flash" in n and "lite" not in n], reverse=True)
            lite = sorted([n for n in found if "flash" in n and "lite" in n], reverse=True)
            pro = sorted([n for n in found if "pro" in n], reverse=True)
            _ACTIVE_GEMINI_MODELS = flash + lite + pro
        except Exception as e:
            print(f"[Startup Gemini list err]: {e}")
        if not _ACTIVE_GEMINI_MODELS:
            _ACTIVE_GEMINI_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"]

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key and not _ACTIVE_GROQ_VISION:
        try:
            r = _HTTP_CLIENT.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {groq_key}"},
            )
            if r.status_code == 200:
                ids = [x["id"] for x in r.json().get("data", [])]
                _ACTIVE_GROQ_VISION = [x for x in ids if "vision" in x or "llama-4" in x]
        except Exception as e:
            print(f"[Startup Groq list err]: {e}")


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
        "gemini_models": _ACTIVE_GEMINI_MODELS[:4],
        "groq_vision_models": _ACTIVE_GROQ_VISION,
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
        if not _ACTIVE_GROQ_VISION:
            raise RuntimeError("No Groq vision models on key")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{PROMPT}\n\n{prompt_text}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }]
        last_err = None
        for mid in _ACTIVE_GROQ_VISION[:2]:
            res = _HTTP_CLIENT.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={"model": mid, "messages": messages, "temperature": 0.0, "max_tokens": 1200},
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
            last_err = f"{mid}:{res.status_code}"
        raise RuntimeError(f"Groq vision failed ({last_err})")
    else:
        res = _HTTP_CLIENT.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "temperature": 0.0,
                "max_tokens": 1200,
            },
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        if res.status_code != 200:
            raise RuntimeError(f"Groq text {res.status_code}")
        return res.json()["choices"][0]["message"]["content"]


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
        last_err = ""

        if _GEMINI_CLIENT:
            for model_name in _ACTIVE_GEMINI_MODELS[:3]:
                try:
                    contents = [PROMPT, contents_text]
                    if img_pil is not None:
                        contents.append(img_pil)
                    cfg = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        max_output_tokens=1500,
                    )
                    res = _GEMINI_CLIENT.models.generate_content(model=model_name, contents=contents, config=cfg)
                    if res and res.text:
                        raw_json = res.text
                        break
                except Exception as e:
                    last_err = f"{model_name}: {str(e)[:60]}"
                    continue

        if not raw_json:
            try:
                raw_json = call_groq(contents_text, img_b64_str)
            except Exception as ge:
                raise RuntimeError(f"{last_err} | {ge}")

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

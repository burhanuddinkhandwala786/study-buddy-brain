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
CACHE_TTL = 900
KEY_COOLDOWN = {}
_DISCOVERED_MODELS = {"list": [], "ts": 0}
_KEY_CYCLE = itertools.count()

PROMPT = """You are an expert academic and technical verification engine.
Verify formulas, option letters, and logical constraints carefully—never be misled by already-clicked radio buttons. Return ONLY valid JSON:
{"detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"UI_GUIDE"|"LONG_ANSWER"|"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"...","pointer_xy":null}
Rules:
- MCQ: cursor_pill = exact option letter/number + text (e.g. "4. Bharuch"). copy_value="", expanded_note="".
- DIRECT_VALUE / CODE_FIX: cursor_pill = exact verified answer/fixed line; copy_value = same raw value; expanded_note="".
- LONG_ANSWER: cursor_pill = short title; copy_value & expanded_note = complete structured breakdown.
- CONCEPT_EXPLAIN: cursor_pill = 1-line summary; expanded_note = 3 high-yield bullet points."""


def get_gemini_keys():
    raw = os.environ.get("GEMINI_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def get_active_gemini_models(client: genai.Client):
    if _DISCOVERED_MODELS["list"] and time.time() - _DISCOVERED_MODELS["ts"] < 3600:
        return _DISCOVERED_MODELS["list"]

    found = []
    try:
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            low = name.lower()
            if "generateContent" in actions and low.startswith("gemini"):
                if not any(b in low for b in ["image", "audio", "tts", "live", "embedding", "gemma", "robotics", "computer-use", "omni", "preview"]):
                    found.append(name)
    except Exception:
        pass

    # Prefer newest active flash models (e.g., gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash)
    flash = sorted([n for n in found if "flash" in n and "lite" not in n], reverse=True)
    lite = sorted([n for n in found if "flash" in n and "lite" in n], reverse=True)
    ordered = flash + lite
    if not ordered:
        ordered = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]

    _DISCOVERED_MODELS["list"] = ordered[:3]
    _DISCOVERED_MODELS["ts"] = time.time()
    return _DISCOVERED_MODELS["list"]


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None


@app.api_route("/", methods=["GET", "HEAD"])
def health():
    keys = get_gemini_keys()
    models = []
    if keys:
        try:
            c = genai.Client(api_key=keys[0], http_options={"timeout": 10000})
            models = get_active_gemini_models(c)
        except Exception:
            pass
    return {
        "status": "ok",
        "provider": "gemini-only-pool",
        "gemini_keys_loaded": len(keys),
        "active_models": models,
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


def call_gemini_only(prompt_text: str, img_pil: Optional[Image.Image] = None) -> str:
    keys = get_gemini_keys()
    if not keys:
        raise RuntimeError("No GEMINI_API_KEY configured in Render")

    now = time.time()
    start_idx = next(_KEY_CYCLE) % len(keys)
    ordered_keys = keys[start_idx:] + keys[:start_idx]
    last_err = "All Gemini keys cooling down — retry in 10s"

    for idx, api_key in enumerate(ordered_keys):
        if KEY_COOLDOWN.get(api_key, 0) > now:
            continue

        client = genai.Client(api_key=api_key, http_options={"timeout": 22000})
        models = get_active_gemini_models(client)

        for model_name in models:
            contents = [PROMPT, prompt_text]
            if img_pil is not None:
                contents.append(img_pil)
            try:
                cfg = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=1000,
                    thinking_config=types.ThinkingConfig(thinking_budget=256),
                )
                res = client.models.generate_content(model=model_name, contents=contents, config=cfg)
                if res and res.text:
                    return res.text
            except Exception as e:
                err_s = str(e)
                if "thinking" in err_s.lower():
                    try:
                        cfg_plain = types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.0,
                            max_output_tokens=1000,
                        )
                        res = client.models.generate_content(model=model_name, contents=contents, config=cfg_plain)
                        if res and res.text:
                            return res.text
                    except Exception as e2:
                        err_s = str(e2)

                last_err = f"Key{idx+1}/{model_name}: {err_s[:65]}"
                if "429" in err_s or "RESOURCE_EXHAUSTED" in err_s:
                    KEY_COOLDOWN[api_key] = time.time() + 30
                    break
                if "403" in err_s or "PERMISSION_DENIED" in err_s or "free tier" in err_s.lower():
                    KEY_COOLDOWN[api_key] = time.time() + 3600
                    break
                continue

    raise RuntimeError(last_err)


@app.post("/analyze")
def analyze_context(req: AnalyzeRequest):
    try:
        contents_text = ""
        img_pil = None
        raw_key = ""

        if req.follow_up and req.prior_context:
            contents_text = f"Context:\n{req.prior_context}\n\nFollow-up:\n{req.follow_up}"
            raw_key = contents_text
        elif req.image_base64:
            clean_b64 = req.image_base64.split(",")[-1]
            raw_key = hashlib.md5(clean_b64.encode()).hexdigest()
            img_pil = Image.open(BytesIO(base64.b64decode(clean_b64))).convert("RGB")
            img_pil.thumbnail((640, 640), Image.BILINEAR)
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

        raw_json = call_gemini_only(contents_text, img_pil)
        data = extract_json(raw_json)

        RESULT_CACHE[key] = (time.time(), data)
        if len(RESULT_CACHE) > 300:
            RESULT_CACHE.pop(next(iter(RESULT_CACHE)))

        return data
    except Exception as e:
        print(f"[GEMINI POOL ERR]: {e}")
        return {
            "detected_type": "ERROR",
            "cursor_pill": "Busy — press hotkey again",
            "copy_value": "",
            "expanded_note": str(e),
            "pointer_xy": None,
        }

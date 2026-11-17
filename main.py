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
CACHE_TTL = 1800  # 30-min cache saves huge quota when users scan similar questions
KEY_COOLDOWN = {}
_KEY_CYCLE = itertools.count()

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

PROMPT = """You are an expert academic and technical verification engine.
Verify formulas, option letters, and logical constraints carefully—never be misled by already-clicked radio buttons. Return ONLY valid JSON:
{"detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"UI_GUIDE"|"LONG_ANSWER"|"CONCEPT_EXPLAIN","cursor_pill":"...","copy_value":"...","expanded_note":"...","pointer_xy":null}
Rules:
- MCQ: cursor_pill = exact option letter/number + text (e.g. "4. Bharuch"). copy_value="", expanded_note="".
- DIRECT_VALUE / CODE_FIX: cursor_pill = exact verified answer/fixed line; copy_value = same raw value; expanded_note="".
- LONG_ANSWER: cursor_pill = short title; copy_value & expanded_note = complete structured breakdown.
- CONCEPT_EXPLAIN: cursor_pill = 1-line summary; expanded_note = 3 high-yield bullet points."""


def get_keys(env_name: str):
    return [k.strip() for k in os.environ.get(env_name, "").split(",") if k.strip()]


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None
    user_api_key: Optional[str] = None  # Supports BYOK in v2


@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {
        "status": "ok",
        "providers": {
            "gemini_keys": len(get_keys("GEMINI_API_KEY")),
            "groq_keys": len(get_keys("GROQ_API_KEY")),
            "nvidia_nim": bool(get_keys("NVIDIA_API_KEY")),
            "openrouter": any(k.startswith("sk-or-") for k in get_keys("OPENROUTER_API_KEY")),
        },
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


def call_gemini_pool(prompt_text: str, img_pil: Optional[Image.Image] = None, extra_key: Optional[str] = None) -> str:
    keys = ([extra_key] if extra_key else []) + get_keys("GEMINI_API_KEY")
    if not keys:
        raise RuntimeError("No Gemini keys")

    now = time.time()
    start_idx = next(_KEY_CYCLE) % len(keys)
    ordered_keys = keys[start_idx:] + keys[:start_idx]
    last_err = "Gemini cooling down"

    for idx, api_key in enumerate(ordered_keys):
        if KEY_COOLDOWN.get(api_key, 0) > now:
            continue
        client = genai.Client(api_key=api_key, http_options={"timeout": 16000})
        for model_name in GEMINI_MODELS:
            contents = [PROMPT, prompt_text] + ([img_pil] if img_pil is not None else [])
            try:
                cfg = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=1000,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                )
                res = client.models.generate_content(model=model_name, contents=contents, config=cfg)
                if res and res.text:
                    return res.text
            except Exception as e:
                err_s = str(e)
                if "thinking" in err_s.lower() or "invalid_argument" in err_s.lower():
                    try:
                        res = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                temperature=0.0,
                                max_output_tokens=1000,
                            ),
                        )
                        if res and res.text:
                            return res.text
                    except Exception as e2:
                        err_s = str(e2)

                last_err = f"G{idx+1}/{model_name}: {err_s[:50]}"
                if "429" in err_s or "RESOURCE_EXHAUSTED" in err_s:
                    KEY_COOLDOWN[api_key] = time.time() + 35
                    break
                if "403" in err_s or "PERMISSION_DENIED" in err_s:
                    KEY_COOLDOWN[api_key] = time.time() + 3600
                    break
                continue
    raise RuntimeError(last_err)


def call_openai_compat(base_url: str, api_key: str, model_id: str, prompt_text: str, img_b64: Optional[str] = None) -> str:
    if img_b64:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{PROMPT}\n\n{prompt_text}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }]
    else:
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": prompt_text},
        ]

    res = _HTTP_CLIENT.post(
        f"{base_url}/chat/completions",
        json={"model": model_id, "messages": messages, "temperature": 0.0, "max_tokens": 700},
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://cursor-companion.app",
            "X-Title": "Cursor Companion",
        },
    )
    if res.status_code != 200:
        raise RuntimeError(f"HTTP {res.status_code}")
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
            img_pil.thumbnail((640, 640), Image.BILINEAR)

            small_img = img_pil.copy()
            small_img.thumbnail((420, 420), Image.BILINEAR)
            buf = BytesIO()
            small_img.save(buf, format="JPEG", quality=62)
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

        # 1. Text fast-path -> Groq (preserves Gemini quota for images)
        if img_pil is None:
            for gkey in get_keys("GROQ_API_KEY"):
                try:
                    raw_json = call_openai_compat("https://api.groq.com/openai/v1", gkey, "llama-3.3-70b-versatile", contents_text)
                    break
                except Exception as e:
                    errors.append(f"Groq-text: {e}")

        # 2. Gemini 3.5-Flash-Lite / 3.8-Flash Key Pool
        if not raw_json:
            try:
                raw_json = call_gemini_pool(contents_text, img_pil, extra_key=req.user_api_key)
            except Exception as e:
                errors.append(str(e))

        # 3. Groq Vision Backup
        if not raw_json:
            for gkey in get_keys("GROQ_API_KEY"):
                try:
                    raw_json = call_openai_compat("https://api.groq.com/openai/v1", gkey, "qwen/qwen3.8-27b", contents_text, img_b64_str)
                    break
                except Exception as e:
                    errors.append(f"Groq-vis: {e}")

        # 4. NVIDIA NIM Free Backup (build.nvidia.com)
        if not raw_json:
            for nkey in get_keys("NVIDIA_API_KEY"):
                try:
                    n_model = "meta/llama-3.2-90b-vision-instruct" if img_b64_str else "meta/llama-3.3-70b-instruct"
                    raw_json = call_openai_compat("https://integrate.api.nvidia.com/v1", nkey, n_model, contents_text, img_b64_str)
                    break
                except Exception as e:
                    errors.append(f"NVIDIA: {e}")

        # 5. OpenRouter Free Models Backup
        if not raw_json:
            for okey in [k for k in get_keys("OPENROUTER_API_KEY") if k.startswith("sk-or-")]:
                try:
                    o_model = "nvidia/nemotron-3.5-super-vl:free" if img_b64_str else "nvidia/nemotron-3-ultra-550b-a55b:free"
                    raw_json = call_openai_compat("https://openrouter.ai/api/v1", okey, o_model, contents_text, img_b64_str)
                    break
                except Exception as e:
                    errors.append(f"OR: {e}")

        if not raw_json:
            raise RuntimeError(" | ".join(errors))

        data = extract_json(raw_json)
        RESULT_CACHE[key] = (time.time(), data)
        if len(RESULT_CACHE) > 500:
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

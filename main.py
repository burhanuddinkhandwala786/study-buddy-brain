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
IMAGE_CACHE = {}
QUESTION_CACHE = {}
CACHE_TTL = 3600
KEY_COOLDOWN = {}
_KEY_CYCLE = itertools.count()

# Verified active models — accuracy-first order
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]

GROQ_TEXT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]

PROMPT = """You are an elite academic, competitive-exam, and engineering verification engine.
Solve step-by-step internally before outputting JSON:
1. Read the ENTIRE question carefully (watch for NOT, INCORRECT, EXCEPT, and unit conversions).
2. Independently verify the factual or mathematical answer BEFORE looking at any highlighted/selected radio buttons.
3. Match your verified answer to the exact option letter/number and text.

Return ONLY valid JSON:
{"detected_type":"MCQ"|"DIRECT_VALUE"|"CODE_FIX"|"UI_GUIDE"|"LONG_ANSWER"|"CONCEPT_EXPLAIN","normalized_q":"lowercase core question text without option letters","cursor_pill":"...","copy_value":"...","expanded_note":"...","pointer_xy":null}
Rules:
- MCQ: cursor_pill = exact option letter/number + text (e.g. "B. Sports"). copy_value="", expanded_note="".
- DIRECT_VALUE / CODE_FIX: cursor_pill = exact verified answer/fixed line; copy_value = same raw value; expanded_note="".
- LONG_ANSWER: cursor_pill = short title; copy_value & expanded_note = complete structured breakdown.
- CONCEPT_EXPLAIN: cursor_pill = 1-line summary; expanded_note = 3 high-yield bullet points."""


def get_keys(env_name: str):
    return [k.strip() for k in os.environ.get(env_name, "").split(",") if k.strip()]


def norm_hash(text: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    return hashlib.md5(clean.encode()).hexdigest() if len(clean) >= 12 else ""


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None
    user_api_key: Optional[str] = None


@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {
        "status": "ok",
        "engine": "v2-precision-3.8",
        "gemini_keys": len(get_keys("GEMINI_API_KEY")),
        "groq_keys": len(get_keys("GROQ_API_KEY")),
        "cached_questions": len(QUESTION_CACHE),
    }


@app.get("/diag")
def diagnose_keys():
    """Live health probe for every Gemini key + Groq key in Render."""
    report = {"gemini": [], "groq": "not_configured"}
    for i, k in enumerate(get_keys("GEMINI_API_KEY"), 1):
        try:
            # Minimum allowed deadline by Google API is >= 10s (15000ms used here)
            c = genai.Client(api_key=k, http_options={"timeout": 15000})
            res = c.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents="Reply OK",
                config=types.GenerateContentConfig(max_output_tokens=10),
            )
            report["gemini"].append({"key": f"G{i} (...{k[-4:]})", "status": "HEALTHY", "reply": (res.text or "").strip()})
        except Exception as e:
            report["gemini"].append({"key": f"G{i} (...{k[-4:]})", "status": "ERROR", "reason": str(e)[:90]})

    gkeys = get_keys("GROQ_API_KEY")
    if gkeys:
        try:
            r = _HTTP_CLIENT.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={"model": GROQ_TEXT_MODELS[0], "messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 10},
                headers={"Authorization": f"Bearer {gkeys[0]}", "Content-Type": "application/json"},
            )
            report["groq"] = "HEALTHY" if r.status_code == 200 else f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            report["groq"] = f"ERROR: {str(e)[:60]}"
    return report


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


def call_gemini_pool(prompt_text: str, img_pil: Optional[Image.Image] = None, extra_key: Optional[str] = None, deep: bool = False) -> str:
    keys = ([extra_key.strip()] if extra_key and extra_key.strip() else []) + get_keys("GEMINI_API_KEY")
    if not keys:
        raise RuntimeError("No Gemini keys")

    now = time.time()
    start_idx = next(_KEY_CYCLE) % len(keys)
    ordered_keys = keys[start_idx:] + keys[:start_idx]
    last_err = "Gemini cooling down"

    for idx, api_key in enumerate(ordered_keys):
        if KEY_COOLDOWN.get(api_key, 0) > now:
            continue
        client = genai.Client(api_key=api_key, http_options={"timeout": 20000})
        for model_name in GEMINI_MODELS:
            contents = [PROMPT, prompt_text] + ([img_pil] if img_pil is not None else [])
            level = "medium" if deep else ("low" if "3.8" in model_name else "minimal")
            try:
                cfg = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=1100,
                    thinking_config=types.ThinkingConfig(thinking_level=level),
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
                                max_output_tokens=1100,
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
                if "403" in err_s or "PERMISSION_DENIED" in err_s or "free tier" in err_s.lower():
                    KEY_COOLDOWN[api_key] = time.time() + 86400
                    break
                continue
    raise RuntimeError(last_err)


def call_groq(prompt_text: str, img_b64: Optional[str] = None) -> str:
    keys = get_keys("GROQ_API_KEY")
    if not keys:
        raise RuntimeError("No Groq keys")

    models_to_try = ["qwen/qwen3.8-27b"] if img_b64 else GROQ_TEXT_MODELS
    last_err = "Groq busy"

    for gkey in keys:
        for model_id in models_to_try:
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
                "https://api.groq.com/openai/v1/chat/completions",
                json={"model": model_id, "messages": messages, "temperature": 0.0, "max_tokens": 700},
                headers={"Authorization": f"Bearer {gkey}", "Content-Type": "application/json"},
            )
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
            last_err = f"Groq {model_id}: HTTP {res.status_code}"
    raise RuntimeError(last_err)


@app.post("/analyze")
def analyze_context(req: AnalyzeRequest):
    try:
        contents_text = ""
        img_pil = None
        img_b64_str = None
        raw_key = ""
        is_followup = bool(req.follow_up and req.prior_context)

        if is_followup:
            contents_text = f"Context:\n{req.prior_context}\n\nFollow-up:\n{req.follow_up}"
            raw_key = contents_text
        elif req.image_base64:
            clean_b64 = req.image_base64.split(",")[-1]
            raw_key = hashlib.md5(clean_b64.encode()).hexdigest()
            img_pil = Image.open(BytesIO(base64.b64decode(clean_b64))).convert("RGB")
            img_pil.thumbnail((760, 760), Image.LANCZOS)

            small_img = img_pil.copy()
            small_img.thumbnail((420, 420), Image.BILINEAR)
            buf = BytesIO()
            small_img.save(buf, format="JPEG", quality=65)
            img_b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

            contents_text = "Solve the problem shown in this image accurately."
        elif req.text_content:
            contents_text = f"Solve:\n{req.text_content}"
            raw_key = req.text_content
            q_hash = norm_hash(req.text_content)
            if q_hash and q_hash in QUESTION_CACHE and time.time() - QUESTION_CACHE[q_hash][0] < CACHE_TTL:
                return QUESTION_CACHE[q_hash][1]
        else:
            return {"detected_type": "ERROR", "cursor_pill": "Select text or snip area first", "copy_value": "", "expanded_note": "", "pointer_xy": None}

        key = hashlib.md5(raw_key.encode()).hexdigest()
        hit = IMAGE_CACHE.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]

        raw_json = None
        errors = []

        # 1. Text queries -> Groq fast-path (keeps 100% of Gemini quota free for image scans)
        if img_pil is None:
            try:
                raw_json = call_groq(contents_text, None)
            except Exception as e:
                errors.append(str(e))

        # 2. Primary Engine: Gemini 3.8 Flash -> 3.5 Flash-Lite pool
        if not raw_json:
            try:
                raw_json = call_gemini_pool(contents_text, img_pil, extra_key=req.user_api_key, deep=is_followup)
            except Exception as e:
                errors.append(str(e))

        # 3. Vision Failover: Groq qwen3.8-27b
        if not raw_json and img_b64_str:
            try:
                raw_json = call_groq(contents_text, img_b64_str)
            except Exception as e:
                errors.append(str(e))

        if not raw_json:
            raise RuntimeError(" | ".join(errors))

        data = extract_json(raw_json)
        norm_q = data.pop("normalized_q", "")
        if not is_followup and norm_q:
            q_hash = norm_hash(norm_q)
            if q_hash:
                if q_hash in QUESTION_CACHE and time.time() - QUESTION_CACHE[q_hash][0] < CACHE_TTL:
                    return QUESTION_CACHE[q_hash][1]
                QUESTION_CACHE[q_hash] = (time.time(), data)
                if len(QUESTION_CACHE) > 1000:
                    QUESTION_CACHE.pop(next(iter(QUESTION_CACHE)))

        IMAGE_CACHE[key] = (time.time(), data)
        if len(IMAGE_CACHE) > 500:
            IMAGE_CACHE.pop(next(iter(IMAGE_CACHE)))

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

import os, re, json, base64, time, hashlib
from io import BytesIO
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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

# ---- All free-tier. Free model names rotate often: override via Render env vars, no redeploy of code needed.
def _env_list(name, default):
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]

GEMINI_MODELS = _env_list("GEMINI_MODELS", "gemini-2.5-flash,gemini-2.5-pro,gemini-2.5-flash-lite")
GEMINI_FAST_MODELS = _env_list("GEMINI_FAST_MODELS", "gemini-2.5-flash-lite,gemini-2.5-flash")
GROQ_MODELS = _env_list("GROQ_MODELS", "openai/gpt-oss-120b,qwen/qwen3-32b,llama-3.3-70b-versatile")
OPENROUTER_MODELS = _env_list("OPENROUTER_MODELS", "deepseek/deepseek-r1:free,meta-llama/llama-3.3-70b-instruct:free")

GROQ_BASE = "https://api.groq.com/openai/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

COOLDOWN = {}
RESULT_CACHE, CACHE_TTL = {}, 600
MAX_IMG_SIDE = 1600  # keep text legible (was 560)


class Solution(BaseModel):
    transcription: str   # exact problem text as read from the image
    reasoning: str       # model works here FIRST
    detected_type: Literal["MCQ", "DIRECT_VALUE", "CODE_FIX", "UI_GUIDE", "LONG_ANSWER", "CONCEPT_EXPLAIN"]
    final_answer: str
    copy_value: str
    expanded_note: str
    confidence: int


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None
    mode: Literal["fast", "accurate"] = "accurate"


SYSTEM = """You are an expert tutor and engineer solving exam-style problems.
Rules:
- First transcribe the problem EXACTLY (numbers, units, options, constraints) into `transcription`.
- Solve step by step in `reasoning`, then re-check by an independent method (substitute back, unit check, eliminate other options).
- If unclear or ambiguous, say so in `reasoning` and lower `confidence`. Never invent missing data.
- MCQ: final_answer = exact option label + text, e.g. "2. Stratified". copy_value="" expanded_note="".
- DIRECT_VALUE / CODE_FIX: final_answer = verified value or corrected line; copy_value = same raw value.
- LONG_ANSWER (>=3 marks, derivations, proofs, comparisons): final_answer = short title; copy_value and expanded_note = complete structured answer.
- CONCEPT_EXPLAIN / UI_GUIDE: final_answer = one-line summary; expanded_note = 3 high-yield bullets.
- confidence: 90+ only if you verified the answer twice."""

JSON_HINT = "\nReturn ONLY one JSON object with keys: transcription, reasoning, detected_type, final_answer, copy_value, expanded_note, confidence (integer 0-100)."


# ---------------- helpers ----------------
def prep_image(b64):
    raw = base64.b64decode(b64.split(",")[-1])
    img = Image.open(BytesIO(raw)).convert("RGB")
    if max(img.size) > MAX_IMG_SIDE:
        img.thumbnail((MAX_IMG_SIDE, MAX_IMG_SIDE), Image.LANCZOS)
    return img


def parse_json(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    m = re.search(r"\{.*\}", txt, re.S)
    d = json.loads(m.group(0))
    if isinstance(d, list) and d:
        d = d[0]
    d.setdefault("transcription", "")
    d.setdefault("reasoning", "")
    d.setdefault("detected_type", "DIRECT_VALUE")
    d.setdefault("final_answer", "")
    d.setdefault("copy_value", "")
    d.setdefault("expanded_note", "")
    try:
        d["confidence"] = int(d.get("confidence", 50))
    except Exception:
        d["confidence"] = 50
    if not d["final_answer"]:
        raise ValueError("empty answer")
    return d


def try_models(provider, models, fn):
    last = None
    for m in models:
        k = f"{provider}:{m}"
        if COOLDOWN.get(k, 0) > time.time():
            continue
        try:
            return fn(m)
        except Exception as e:
            last, s = e, str(e).lower()
            if any(x in s for x in ("429", "resource_exhausted", "rate limit", "quota")):
                COOLDOWN[k] = time.time() + 65
            elif any(x in s for x in ("404", "not found", "decommission", "deprecated", "no longer", "does not exist")):
                COOLDOWN[k] = time.time() + 3600
            else:
                COOLDOWN[k] = time.time() + 20
    raise RuntimeError(f"{provider} unavailable: {str(last)[:80]}")


# ---------------- providers ----------------
def call_gemini(user_text, img, temp, models):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("no GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    contents = [user_text] + ([img] if img is not None else [])

    def run(model):
        cfg = dict(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=Solution,
            temperature=temp,
            max_output_tokens=10000,  # thinking tokens count toward this
        )
        if "2.5" in model and "lite" not in model:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=-1 if "pro" in model else 4096)
        try:
            r = client.models.generate_content(model=model, contents=contents,
                                               config=types.GenerateContentConfig(**cfg))
        except Exception as e:
            if "thinking" in str(e).lower() and "thinking_config" in cfg:
                cfg.pop("thinking_config")
                r = client.models.generate_content(model=model, contents=contents,
                                                   config=types.GenerateContentConfig(**cfg))
            else:
                raise
        return parse_json(r.text)

    return try_models("gemini", models, run)


def call_oai(provider, base, key_env, models, user_text, temp=0.0, system=None, json_mode=True):
    key = os.environ.get(key_env, "").strip()
    if not key:
        raise RuntimeError(f"no {key_env}")

    def run(model):
        body = {
            "model": model,
            "temperature": temp,
            "max_tokens": 6000,
            "messages": [
                {"role": "system", "content": (system or SYSTEM) + JSON_HINT},
                {"role": "user", "content": user_text},
            ],
        }
        if json_mode and provider == "groq":
            body["response_format"] = {"type": "json_object"}
        r = httpx.post(base + "/chat/completions", headers={"Authorization": f"Bearer {key}"},
                       json=body, timeout=75)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.text[:150]}")
        return parse_json(r.json()["choices"][0]["message"]["content"])

    return try_models(provider, models, run)


def call_groq(problem, temp=0.0, offset=0):
    models = GROQ_MODELS[offset:] + GROQ_MODELS[:offset]
    return call_oai("groq", GROQ_BASE, "GROQ_API_KEY", models, f"Problem:\n{problem}", temp)


def call_openrouter(problem, temp=0.0):
    return call_oai("openrouter", OPENROUTER_BASE, "OPENROUTER_API_KEY", OPENROUTER_MODELS,
                    f"Problem:\n{problem}", temp, json_mode=False)


CRITIC_SYSTEM = """You are a strict grader. Given a problem and a proposed answer, check it for factual, mathematical or logical errors."""


def critique(problem, answer_text):
    prompt = f"Problem:\n{problem}\n\nProposed answer:\n{answer_text}\n\nIs it fully correct? Put 'OK' in final_answer if yes; otherwise put the specific error and correction. Set confidence for your verdict."
    try:
        d = call_oai("groq", GROQ_BASE, "GROQ_API_KEY", GROQ_MODELS, prompt, 0.0, system=CRITIC_SYSTEM)
        return d["final_answer"].strip()
    except Exception:
        return None


# ---------------- voting ----------------
def norm_key(s):
    a = (s.get("final_answer") or "").strip().lower()
    if s.get("detected_type") == "MCQ":
        m = re.match(r"\(?([a-z0-9])[\)\.\s:]", a + " ")
        return m.group(1) if m else re.sub(r"\W", "", a)
    return re.sub(r"[\s,]", "", a)


def run_parallel(fns):
    out = []
    with ThreadPoolExecutor(max_workers=max(1, len(fns))) as ex:
        futs = [ex.submit(f) for f in fns]
        for f in futs:
            try:
                out.append(f.result(timeout=100))
            except Exception:
                pass
    return out


def majority(cands):
    votes = Counter(norm_key(c) for c in cands)
    top, n = votes.most_common(1)[0]
    best = max((c for c in cands if norm_key(c) == top), key=lambda c: c["confidence"])
    return best, n


def solve(req, img, text_in):
    if req.mode == "fast" and not req.follow_up:
        return call_gemini(text_in, img, 0.0, GEMINI_FAST_MODELS), "single"

    primary = call_gemini(text_in, img, 0.0, GEMINI_MODELS)
    problem = (primary.get("transcription") or text_in).strip()

    if primary["detected_type"] in ("LONG_ANSWER", "CONCEPT_EXPLAIN", "UI_GUIDE"):
        full = primary["copy_value"] or primary["expanded_note"]
        verdict = critique(problem, f"{primary['final_answer']}\n{full}")
        if verdict and verdict.upper() != "OK" and len(verdict) > 3:
            primary["expanded_note"] += f"\n\nReviewer note: {verdict[:400]}"
            primary["confidence"] = min(primary["confidence"], 65)
            return primary, "uncertain"
        return primary, ("verified" if verdict else "single")

    # Short-answer: independent verifiers on the transcribed text
    verifiers = run_parallel([lambda: call_groq(problem, 0.0), lambda: call_openrouter(problem, 0.0)])
    cands = [primary] + verifiers

    if len(cands) >= 3 or (len(cands) == 2):
        best, n = majority(cands)
        if n >= 2:
            if n == len(cands):
                best["confidence"] = max(best["confidence"], 90)
            return best, "verified"

    if len(cands) == 1:  # no verifier available (keys missing / rate limited)
        primary["confidence"] = min(primary["confidence"], 75)
        return primary, "single"

    # Disagreement: bring in extra opinions (second Groq model, second Gemini sample)
    extra = run_parallel([lambda: call_groq(problem, 0.3, offset=1),
                          lambda: call_gemini(text_in, img, 0.5, GEMINI_MODELS)])
    cands += extra
    best, n = majority(cands)
    if n >= 3:
        return best, "verified"
    if n >= 2 and n > len(cands) / 2:
        best["confidence"] = min(best["confidence"], 75)
        return best, "verified"
    best["confidence"] = min(best["confidence"], 50)
    return best, "uncertain"


# ---------------- routes ----------------
@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {
        "status": "ok",
        "gemini": bool(os.environ.get("GEMINI_API_KEY")),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
        "cooling_down": [k for k, v in COOLDOWN.items() if v > time.time()],
    }


def error_response(msg, detail=""):
    return {"detected_type": "ERROR", "cursor_pill": msg, "copy_value": "",
            "expanded_note": detail, "pointer_xy": None}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return error_response("Set GEMINI_API_KEY in Render")
    try:
        img = None
        if req.follow_up and req.prior_context:
            text_in = f"Earlier answer/context:\n{req.prior_context}\n\nFollow-up question:\n{req.follow_up}"
            raw_key = req.prior_context + req.follow_up
        elif req.image_base64:
            img = prep_image(req.image_base64)
            text_in = "Solve the problem shown in this image."
            raw_key = req.image_base64.split(",")[-1]
        elif req.text_content:
            text_in = f"Solve this:\n{req.text_content}"
            raw_key = req.text_content
        else:
            return error_response("Empty input")

        key = hashlib.md5((req.mode + raw_key).encode()).hexdigest()
        hit = RESULT_CACHE.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]

        sol, status = solve(req, img, text_in)
        conf = sol["confidence"]
        pill = sol["final_answer"]
        if status == "uncertain" or conf < 60:
            pill = "⚠ " + pill
        elif status == "verified":
            pill = "✓ " + pill

        data = {
            "detected_type": sol["detected_type"],
            "cursor_pill": pill,
            "copy_value": sol["copy_value"],
            "expanded_note": sol["expanded_note"],
            "pointer_xy": None,
            "confidence": conf,
            "status": status,
        }
        if status != "uncertain":
            RESULT_CACHE[key] = (time.time(), data)
            if len(RESULT_CACHE) > 200:
                RESULT_CACHE.pop(next(iter(RESULT_CACHE)))
        return data
    except Exception as e:
        return error_response(f"Err: {str(e)[:55]}", str(e))

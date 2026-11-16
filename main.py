import os, re, json, base64, time, hashlib
from io import BytesIO
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Literal

from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

try:
    import anthropic
except ImportError:
    anthropic = None

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Override in Render env vars without redeploying code
GEMINI_STRONG = os.environ.get("GEMINI_STRONG", "gemini-2.5-pro")
GEMINI_FAST = os.environ.get("GEMINI_FAST", "gemini-2.5-flash")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
CLAUDE_STRONG = os.environ.get("CLAUDE_STRONG", "claude-opus-5-5")

RESULT_CACHE, CACHE_TTL = {}, 600
MAX_IMG_SIDE = 1600  # do NOT shrink to 560


class Solution(BaseModel):
    reasoning: str            # model must work here FIRST
    detected_type: Literal["MCQ", "DIRECT_VALUE", "CODE_FIX", "UI_GUIDE", "LONG_ANSWER", "CONCEPT_EXPLAIN"]
    final_answer: str         # MCQ: "2. Stratified" | value | fixed line | short title
    copy_value: str
    expanded_note: str
    confidence: int           # 0-100, honest self-assessment


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None
    follow_up: Optional[str] = None
    prior_context: Optional[str] = None
    mode: Literal["fast", "accurate"] = "accurate"


SYSTEM = """You are an expert tutor and engineer solving exam-style problems.
Rules:
- First read the problem EXACTLY as shown: transcribe all numbers, units, options and constraints into `reasoning`.
- Solve step by step in `reasoning`, then re-check the result by an independent method (substitute back, unit check, eliminate other options).
- If the image is unclear or the question is ambiguous, say so in `reasoning` and lower `confidence`. Never invent missing data.
- MCQ: final_answer = exact option label + text, e.g. "2. Stratified". copy_value="" expanded_note="".
- DIRECT_VALUE / CODE_FIX: final_answer = verified value or corrected line; copy_value = same raw value.
- LONG_ANSWER (>=3 marks, derivations, proofs, comparisons): final_answer = short title; copy_value and expanded_note = complete structured answer.
- CONCEPT_EXPLAIN / UI_GUIDE: final_answer = one-line summary; expanded_note = 3 high-yield bullets.
- confidence: 90+ only if you verified the answer twice."""


# ---------- input prep ----------
def prep_image(b64: str):
    raw = base64.b64decode(b64.split(",")[-1])
    img = Image.open(BytesIO(raw)).convert("RGB")
    if max(img.size) > MAX_IMG_SIDE:
        img.thumbnail((MAX_IMG_SIDE, MAX_IMG_SIDE), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, "PNG")  # lossless keeps small text crisp
    return img, base64.b64encode(buf.getvalue()).decode()


# ---------- model callers ----------
def call_gemini(model, user_text, img, temp):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip())
    contents = [user_text] + ([img] if img is not None else [])
    cfg = dict(
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_schema=Solution,
        temperature=temp,
        max_output_tokens=12000,  # thinking tokens count against this
    )
    if "2.5" in model:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=-1 if "pro" in model else 4096)
    r = client.models.generate_content(model=model, contents=contents, config=types.GenerateContentConfig(**cfg))
    return json.loads(r.text)


def call_claude(model, user_text, img_b64, temp):
    if anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("claude unavailable")
    c = anthropic.Anthropic()
    content = []
    if img_b64:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}})
    content.append({"type": "text", "text": user_text})
    r = c.messages.create(
        model=model, max_tokens=8000, temperature=temp,
        system=SYSTEM + "\nReturn ONLY one JSON object with keys: reasoning, detected_type, final_answer, copy_value, expanded_note, confidence.",
        messages=[{"role": "user", "content": content}],
    )
    txt = "".join(b.text for b in r.content if b.type == "text")
    return json.loads(re.search(r"\{.*\}", txt, re.S).group(0))


# ---------- voting ----------
def norm_key(s: dict) -> str:
    a = (s.get("final_answer") or "").strip().lower()
    if s.get("detected_type") == "MCQ":
        m = re.match(r"\(?([a-z0-9])[\)\.\s:]", a + " ")
        return m.group(1) if m else re.sub(r"\W", "", a)
    return re.sub(r"[\s,]", "", a)


def run_all(jobs):
    out = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = [ex.submit(fn, *args) for fn, args in jobs]
        for f in futs:
            try:
                out.append(f.result(timeout=90))
            except Exception:
                pass
    return out


def solve(user_text, img, img_b64, mode, follow_up):
    if mode == "fast" and not follow_up:
        return call_gemini(GEMINI_FAST, user_text, img, 0.0), "single"

    jobs = [
        (call_gemini, (GEMINI_STRONG, user_text, img, 0.0)),
        (call_gemini, (GEMINI_STRONG, user_text, img, 0.5)),
        (call_claude, (CLAUDE_MODEL, user_text, img_b64, 0.0)),
    ]
    sols = run_all(jobs)
    if not sols:
        raise RuntimeError("All models failed")

    long_form = sols[0].get("detected_type") in ("LONG_ANSWER", "CONCEPT_EXPLAIN", "UI_GUIDE")
    if long_form:
        best = max(sols, key=lambda s: s.get("confidence", 0))
        return best, "single"

    votes = Counter(norm_key(s) for s in sols)
    top, n = votes.most_common(1)[0]
    if n >= 2:
        best = max((s for s in sols if norm_key(s) == top), key=lambda s: s.get("confidence", 0))
        best["confidence"] = max(best.get("confidence", 0), 90 if n == len(sols) else 80)
        return best, "verified"

    # Disagreement -> tiebreaker with strongest models
    tie = run_all([
        (call_claude, (CLAUDE_STRONG, user_text, img_b64, 0.0)),
        (call_gemini, (GEMINI_STRONG, user_text, img, 0.0)),
    ])
    sols += tie
    votes = Counter(norm_key(s) for s in sols)
    top, n = votes.most_common(1)[0]
    best = max((s for s in sols if norm_key(s) == top), key=lambda s: s.get("confidence", 0))
    if n >= 3:
        return best, "verified"
    best["confidence"] = min(best.get("confidence", 50), 55)
    return best, "uncertain"


# ---------- routes ----------
@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "strong": GEMINI_STRONG, "fast": GEMINI_FAST, "claude": bool(anthropic and os.environ.get("ANTHROPIC_API_KEY"))}


def error_response(msg, detail=""):
    return {"detected_type": "ERROR", "cursor_pill": msg, "copy_value": "", "expanded_note": detail, "pointer_xy": None}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return error_response("Set GEMINI_API_KEY in Render")
    try:
        img = img_b64 = None
        if req.follow_up and req.prior_context:
            user_text = f"Earlier answer/context:\n{req.prior_context}\n\nFollow-up question:\n{req.follow_up}"
            key = req.prior_context + req.follow_up
        elif req.image_base64:
            img, img_b64 = prep_image(req.image_base64)
            user_text = "Solve the problem shown in this image."
            key = img_b64
        elif req.text_content:
            user_text = f"Solve this:\n{req.text_content}"
            key = req.text_content
        else:
            return error_response("Empty input")

        key = hashlib.md5((req.mode + key).encode()).hexdigest()
        hit = RESULT_CACHE.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]

        sol, status = solve(user_text, img, img_b64, req.mode, bool(req.follow_up))
        conf = int(sol.get("confidence", 0))
        pill = sol["final_answer"]
        if status == "uncertain" or conf < 60:
            pill = "⚠ " + pill
        elif status == "verified":
            pill = "✓ " + pill

        data = {
            "detected_type": sol["detected_type"],
            "cursor_pill": pill,
            "copy_value": sol.get("copy_value", ""),
            "expanded_note": sol.get("expanded_note", ""),
            "pointer_xy": None,
            "confidence": conf,
            "status": status,
        }
        if status != "uncertain":  # never cache doubtful answers
            RESULT_CACHE[key] = (time.time(), data)
            if len(RESULT_CACHE) > 200:
                RESULT_CACHE.pop(next(iter(RESULT_CACHE)))
        return data
    except Exception as e:
        return error_response(f"Err: {str(e)[:55]}", str(e))

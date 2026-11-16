import os
import json
import base64
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

class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    text_content: Optional[str] = None

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze_context(req: AnalyzeRequest):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {
            "detected_type": "ERROR",
            "cursor_pill": "Set GEMINI_API_KEY in Render",
            "copy_value": "",
            "expanded_note": "",
            "pointer_xy": None
        }

    try:
        client = genai.Client(api_key=api_key)

        prompt = """
        You are an ultra-fast, context-aware cursor intelligence. Analyze the input strictly by context and return valid JSON with keys:
        - "detected_type": "MCQ" | "DIRECT_VALUE" | "CODE_FIX" | "UI_GUIDE" | "CONCEPT_EXPLAIN"
        - "cursor_pill": Ultra-short answer displayed directly on the user's cursor.
          * If MCQ: ONLY the exact option letter and text (e.g., "B) Half-wave dipole"). Zero explanation.
          * If Numerical / Fill-in-blank: ONLY the exact final value/unit.
          * If Code error: ONLY the corrected line of code.
          * If UI/Software screen asking how/where to do something: 1-line instruction (e.g., "Click Export → PDF").
          * If Open-ended theory/diagram: 1 crisp sentence core takeaway.
        - "copy_value": The clean raw value/code/answer to auto-copy to clipboard (empty string for MCQs).
        - "expanded_note": If CONCEPT_EXPLAIN or UI_GUIDE, give a concise 3-bullet step-by-step guide; otherwise "".
        - "pointer_xy": If UI_GUIDE and a specific button/icon is visible in the image, return [x, y] normalized from 0 to 1000; otherwise null.
        """

        contents = [prompt]
        if req.image_base64:
            clean_b64 = req.image_base64.split(",")[-1]
            img = Image.open(BytesIO(base64.b64decode(clean_b64))).convert("RGB")
            contents.append(img)
        elif req.text_content:
            contents.append(f"Selected text:\n{req.text_content}")
        else:
            return {"detected_type": "ERROR", "cursor_pill": "Empty input", "copy_value": "", "expanded_note": "", "pointer_xy": None}

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "detected_type": "ERROR",
            "cursor_pill": f"Err: {str(e)[:50]}",
            "copy_value": "",
            "expanded_note": str(e),
            "pointer_xy": None
        }

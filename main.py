import os
import json
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScreenRequest(BaseModel):
    image_base64: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze_context(req: ScreenRequest):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {
            "detected_type": "ERROR",
            "cursor_pill": "Set GEMINI_API_KEY in Render",
            "expanded_note": ""
        }

    try:
        client = genai.Client(api_key=api_key)
        clean_b64 = req.image_base64.split(",")[-1]
        img = Image.open(BytesIO(base64.b64decode(clean_b64))).convert("RGB")

        prompt = """
        Analyze the image strictly by context and return valid JSON with keys:
        - "detected_type": "MCQ" | "DIRECT_VALUE" | "CODE_FIX" | "CONCEPT_EXPLAIN"
        - "cursor_pill": Ultra-short direct answer shown on cursor.
          * If MCQ: ONLY the exact option letter and text (e.g., "B) Half-wave dipole"). Zero explanation.
          * If Numerical / Fill-in-blank: ONLY the exact final value/unit.
          * If Code error: ONLY the fixed line of code.
          * If Open-ended theory/diagram: 1 crisp sentence summary.
        - "expanded_note": If CONCEPT_EXPLAIN, give a 3-bullet step-by-step explanation; otherwise "".
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "detected_type": "ERROR",
            "cursor_pill": f"API Err: {str(e)[:55]}",
            "expanded_note": str(e)
        }

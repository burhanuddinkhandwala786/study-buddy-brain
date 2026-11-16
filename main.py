import os
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

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class ContextResponse(BaseModel):
    detected_type: str   # "MCQ", "DIRECT_VALUE", "CODE_FIX", or "CONCEPT_EXPLAIN"
    cursor_pill: str     # Ultra-compact answer shown right on the cursor (e.g., "C) 14.2 dB")
    expanded_note: str   # Short 2-3 line breakdown if the user hovers/expands the cursor

class ScreenRequest(BaseModel):
    image_base64: str

@app.post("/analyze", response_model=ContextResponse)
def analyze_context(req: ScreenRequest):
    clean_b64 = req.image_base64.split(",")[-1]
    img = Image.open(BytesIO(base64.b64decode(clean_b64)))

    system_rules = """
    You are an ultra-fast, context-aware cursor intelligence. Analyze the image strictly by context:
    1. MCQ / Multiple Choice: Set detected_type="MCQ". Put ONLY the exact option letter and answer text in cursor_pill (e.g., "B) Half-wave dipole"). Zero explanation in cursor_pill.
    2. Fill-in-the-blank / Math / Numerical: Set detected_type="DIRECT_VALUE". Put ONLY the final exact number/unit/word in cursor_pill.
    3. Code / Terminal Error: Set detected_type="CODE_FIX". Put ONLY the corrected line of code in cursor_pill.
    4. Open-ended Theory / Diagram: Set detected_type="CONCEPT_EXPLAIN". Put a crisp 1-sentence core takeaway in cursor_pill, and a concise 3-bullet step-by-step summary in expanded_note.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[system_rules, img],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ContextResponse,
            temperature=0.1,
        ),
    )
    return response.parsed

import os
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

app = FastAPI()

# Allow your desktop app to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reads your secret key from Render's environment settings
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class ScreenRequest(BaseModel):
    image_base64: str

@app.get("/")
def home():
    return {"status": "Study Buddy Brain is awake!"}

@app.post("/explain")
def explain_screen(req: ScreenRequest):
    # Clean the base64 image string and turn it into an image
    clean_b64 = req.image_base64.split(",")[-1]
    img_bytes = base64.b64decode(clean_b64)
    img = Image.open(BytesIO(img_bytes))

    prompt = (
        "You are a friendly, smart study buddy. Look at the selected question, code, "
        "or diagram in this image. Explain the core concept first, walk through the "
        "solution step-by-step clearly, and end with one quick check-question to help "
        "the student master it. Keep formatting clean and concise."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img]
    )
    return {"explanation": response.text}
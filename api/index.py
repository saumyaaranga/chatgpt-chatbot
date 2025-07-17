from flask import Flask, request, jsonify, render_template
import openai as openai_router
from google import genai
from google.genai import types
import base64
import time
import os
from flask_cors import CORS

# 🔐 API KEYS from environment (for Vercel)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GENAI_API_KEY = os.environ.get("GENAI_API_KEY", "")

# OpenRouter setup
openai_router_client = openai_router.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Gemini client setup
genai_client = genai.Client(api_key=GENAI_API_KEY)

# Flask app
app = Flask(__name__)
CORS(app)

# 🎯 GPT helper
def ask_gpt(prompt, model="openai/gpt-4.1-nano"):
    try:
        response = openai_router_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            timeout=40
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT Error]: {e}"

# 🤖 DeepSeek helper
def ask_deepseek(prompt):
    try:
        response = openai_router_client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            timeout=40
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[DeepSeek Error]: {e}"

# 🖼️ Gemini-based image generation
def generate_image(prompt):
    print(f"🎨 Gemini image prompt: {prompt}")
    try:
        resp = genai_client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
        )
        # Extract image byte data
        for part in resp.candidates[0].content.parts:
            if part.inline_data is not None:
                return base64.b64encode(part.inline_data.data).decode()
        return None
    except Exception as e:
        print("❌ Gemini image error:", e)
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def handle_question():
    q = request.json.get("question", "").strip()
    if not q:
        return jsonify({"error": "No question provided"}), 400

    start = time.time()

    better = ask_gpt(f"Improve clarity: {q}")
    gpt_ans = ask_gpt(f"Answer in detail:\n{better}")
    ds_ans = ask_deepseek(f"Also answer:\n{better}")

    merged = ask_gpt(
        f"The user asked: {q}\n\nGPT says:\n{gpt_ans}\n\nDeepSeek says:\n{ds_ans}\n\nNow combine into a clear article (markdown)."
    )

    visual_prompts = ask_gpt(
        f"From the article below, generate *two* vivid image prompts (1 sentence each). Article:\n{merged}"
    ).split("\n")

    # Clean prompts and get images
    images = []
    for p in [p.strip() for p in visual_prompts if p.strip()][:2]:
        img_b64 = generate_image(p)
        images.append({"prompt": p, "base64": img_b64 or ""})

    return jsonify({
        "final": merged,
        "chatgpt_answer": gpt_ans,
        "deepseek_answer": ds_ans,
        "images": images
    })

# ✅ Required by Vercel — exposes the Flask app
app = app

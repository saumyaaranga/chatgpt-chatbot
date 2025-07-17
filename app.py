from flask import Flask, request, jsonify, render_template
import openai as openai_router
from google import genai
from google.genai import types
import base64
import time
from flask_cors import CORS

# 🔐 API KEYS
OPENROUTER_API_KEY = "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"
GENAI_API_KEY = "AIzaSyAmqlLAHoDRVll5mTT7GEyCm_jHKSvwAMo"

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
            config=types.GenerateContentConfig(response_modalities=["TEXT","IMAGE"])
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
    images = []
    for p in visual_prompts[:2]:
        img_b64 = generate_image(p.strip())
        images.append({"prompt": p.strip(), "base64": img_b64 or ""})

    return jsonify({
        "final": merged,
        "chatgpt_answer": gpt_ans,
        "deepseek_answer": ds_ans,
        "images": images
    })

if __name__ == "__main__":
    app.run(debug=True)



# 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
# http://127.0.0.1:5000/
#openai/gpt-4.1-nano
#deepseek/deepseek-r1-0528:free
# AIzaSyAmqlLAHoDRVll5mTT7GEyCm_jHKSvwAMo
from flask import Flask, request, jsonify, render_template
import os
import openai
import google.generativeai as genai  # ✅ Correct import
from google.generativeai import types  # ✅ Correct import
import base64
import time
from flask_cors import CORS

# ✅ Setup Flask
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))
CORS(app)
print("🔥 Flask app is starting...")
# ✅ Load environment variables
OPENROUTER_API_KEY ="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"
GENAI_API_KEY ="AIzaSyAmqlLAHoDRVll5mTT7GEyCm_jHKSvwAMo"

# ✅ Configure OpenRouter client
openai_router_client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ✅ Configure Gemini client
genai.configure(api_key=GENAI_API_KEY)


# ✅ GPT Helper
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

# ✅ DeepSeek Helper
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

# ✅ Gemini Image Generation
def generate_image(prompt):
    print(f"🎨 Gemini image prompt: {prompt}")
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-preview-image-generation")
        resp = model.generate_content(
            prompt,
            generation_config=types.GenerationConfig(response_modality=["IMAGE"])
        )
        for part in resp.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                return base64.b64encode(part.inline_data.data).decode()
        return None
    except Exception as e:
        print("❌ Gemini image error:", e)
        return None


# ✅ Route: Homepage
@app.route("/")
def home():
    return render_template("index.html")

# ✅ Route: /ask
@app.route("/ask", methods=["POST"])
def handle_question():
    try:
        q = request.json.get("question", "").strip()
        if not q:
            return jsonify({"error": "No question provided"}), 400

        # 1. Improve question
        better = ask_gpt(f"Improve clarity: {q}")
        # 2. Get answers
        gpt_ans = ask_gpt(f"Answer in detail:\n{better}")
        ds_ans = ask_deepseek(f"Also answer:\n{better}")
        # 3. Merge
        merged = ask_gpt(
            f"The user asked: {q}\n\nGPT says:\n{gpt_ans}\n\nDeepSeek says:\n{ds_ans}\n\nNow combine into a clear article (markdown)."
        )
        # 4. Get image prompts
        visual_prompts = ask_gpt(
            f"From the article below, generate *two* vivid image prompts (1 sentence each). Article:\n{merged}"
        ).split("\n")

        # 5. Generate images
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/test")
def test():
    return "✅ Flask is working on Vercel!"
if __name__ == "__main__":
    app.run(debug=True)


# ✅ NOTE: No app.run() here — Vercel auto-handles it

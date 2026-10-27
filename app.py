import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)  # 🔓 បើកទ្វារ CORS

# ប្រើប្រាស់ Model 0.5B
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("កំពុងផ្ទុក Model ចូលក្នុង CPU RAM... សូមរង់ចាំបន្តិច")

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map=None
)
model.to(device)

# --- មុខងារសម្រាប់អាន Dataset ចอกจาก Folder "data/" ---
DATA_DIR = "data"

def load_dataset_context(user_query=""):
    short_greetings = ["hi", "hello", "hey", "សួស្តី", "ជម្រាបសួរ", "bye", "សុខសប្បាយជាទេ"]
    clean_query = user_query.strip().lower()
    if len(clean_query) < 8 or clean_query in short_greetings:
        return ""

    context = ""
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".txt"):
                file_path = os.path.join(DATA_DIR, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        context += f"\n--- Document ({filename}) ---\n{content}\n"
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    return context

print("Model និង RAG System ត្រូវបានរៀបចំរួចរាល់នៅលើ CPU!")

# 🌐 [NEW]: Route សម្រាប់បង្ហាញ UI (index.html) នៅទំព័រដើម
@app.route("/")
def home():
    return render_template("index.html")

# 💬 [NEW]: Route សម្រាប់รับសំណួរពី UI មកកាន់ AI
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "គ្មានសារត្រូវបានបញ្ចូល"}), 400

    dataset_context = load_dataset_context(user_message)

    if dataset_context:
        system_prompt = (
            "You are Meng AI, a precise local assistant. "
            "Answer the user's question accurately using ONLY the Knowledge Base below. "
            "If the answer is not in the Knowledge Base, say 'ខ្ញុំមិនមានទិន្នន័យនេះទេ។'\n\n"
            f"=== KNOWLEDGE BASE ===\n{dataset_context}\n======================"
        )
    else:
        system_prompt = (
            "You are Meng AI, an elite-level technical and logical reasoning assistant. "
            "Always break down complex problems step-by-step before answering."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    text_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,   
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.15
        )

    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)  
# Load  Flan-T5-model
model = pipeline("text2text-generation", model="Rohith630/flan-t5-customer-chatbot")
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    question = data.get("question", "")
    result = model(
        question,
        max_length=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id
    )
    return jsonify({"response": result[0]["generated_text"]})



if __name__ == "__main__":
    app.run(debug=True, port=5000)

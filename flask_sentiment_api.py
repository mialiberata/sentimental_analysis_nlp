from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Carregar modelo e tokenizador
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Mapeamento de classes
label_dict = {0: "negativo", 1: "neutro", 2: "positivo"}

@app.route("/", methods=["GET"])
def home():
    return "<h1>API de Análise de Sentimento</h1><p>Envie um POST para <b>/predict</b> com um texto para análise.</p>"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"erro": "Texto não fornecido"}), 400
    
    encoding = tokenizer(
        text, truncation=True, padding='max_length', max_length=128, return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        pred_label = torch.argmax(outputs.logits, axis=1).item()
    
    return jsonify({"sentimento": label_dict[pred_label]})

# Tratamento de erro 404 (Página não encontrada)
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"erro": "Página não encontrada. Por favor, verifique o endereço e tente novamente."}), 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

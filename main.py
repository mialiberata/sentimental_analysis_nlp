import pandas as pd
import re
import nltk
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels

# Baixar recursos necessários do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Dispositivo (GPU se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Expansão de postagens sobre marcas
data = {
    "text": [
        "Amo a Coca-Cola! Sempre me dá energia!", 
        "Meta está arruinando a privacidade online, que horror!",
        "A Tesla inova muito, mas os preços são absurdos!",
        "O Google sabe tudo sobre mim, isso me assusta um pouco.",
        "Não gosto do sabor da Coca-Cola, prefiro Pepsi!",
        "Meta está investindo bem no metaverso, gostei!",
        "Os carros da Tesla são lindos, quero um!",
        "O Google melhorou muito suas buscas, incrível!",
        "Coca-Cola sempre me traz boas lembranças, adoro!",
        "O Meta está virando um caos, cheio de anúncios irrelevantes.",
        "Tesla tem uma tecnologia incrível, mas o suporte ao cliente precisa melhorar.",
        "Google facilita minha vida todos os dias, sou fã!",
        "Amo o sabor da Coca-Cola gelada!", 
        "Meta deveria se preocupar mais com segurança digital!",
        "Tesla é o futuro dos carros elétricos!",
        "O Google faz parte da minha rotina, não consigo viver sem!",
        "A Tesla prometeu atualizações, mas ainda não lançou nada!",
        "Meta censura conteúdos de forma injusta, um absurdo!",
        "A experiência com o suporte do Google foi péssima!",
        "Coca-Cola é puro açúcar e faz mal à saúde!"
    ],
    "sentiment": [
        "positivo", "negativo", "negativo", "neutro", "negativo", "positivo", "positivo", "positivo", 
        "positivo", "negativo", "neutro", "positivo", "positivo", "negativo", "positivo", "positivo", 
        "negativo", "negativo", "negativo", "negativo"
    ]
}

# Criando DataFrame
df = pd.DataFrame(data)

# Separando dados
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Mapeamento de classes
label_dict = {"negativo": 0, "neutro": 1, "positivo": 2}
df["label"] = df["sentiment"].map(label_dict)

# Tokenizador BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Criando datasets
dataset_train = SentimentDataset(X_train.tolist(), [label_dict[y] for y in y_train], tokenizer)
dataset_test = SentimentDataset(X_test.tolist(), [label_dict[y] for y in y_test], tokenizer)

dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=4)

# Carregar modelo pré-treinado BERT
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(device)

# Otimizador e Loss Function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Treinamento do modelo
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader_train:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Época {epoch+1}, Loss: {total_loss/len(dataloader_train)}")

# Avaliação do modelo
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in dataloader_test:
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, axis=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Ajustando o relatório de classificação
labels_presentes = unique_labels(true_labels, predictions)
labels_mapeadas = [list(label_dict.keys())[i] for i in labels_presentes]

print("Acurácia:", accuracy_score(true_labels, predictions))
print("Relatório de Classificação:\n", classification_report(true_labels, predictions, target_names=labels_mapeadas))

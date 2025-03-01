# 🔎 Análise de Sentimento com BERT

Oi, Mia aqui! ִֶָ𓂃 ࣪˖ ִֶָ🐇་༘࿐
Se tem uma coisa que me fascina, é como palavras carregam camadas de significado. ⸜(｡˃ ᵕ ˂ )⸝♡
Algumas marcas despertam paixão, outras irritação, e às vezes a gente só segue a vida sem se importar muito, mas como isso se traduz na linguagem? 
Como verbos e adjetivos revelam nossas percepções sobre empresas gigantes como Google, Tesla, Coca-Cola e Meta?

Este projeto é exatamente sobre isso. Peguei um modelo **BERT**, treinei com um dataset de frases sobre essas marcas e criei uma API para analisar **se um texto expressa sentimento positivo, negativo ou neutro**. Bora ver como funciona? 👀

## 🎯 **O que eu queria com isso?** 

Mais do que um classificador de sentimentos qualquer, eu quero entender **o que gera cada emoção**. Não basta saber que alguém “ama” ou “odeia” uma marca, é preciso entender **por quê**. E aí entra a importância dos **verbos e adjetivos** usados. A estrutura da frase conta muito!

## 🏢 **Por que essas marcas?**

A escolha não foi aleatória. Essas empresas dominam nossas interações diárias e, justamente por isso, despertam todo tipo de sentimento:

- **Coca-Cola** 🥤 — Associada a nostalgia, prazer e questões de saúde.
- **Meta (Facebook, Instagram, WhatsApp)** 📱 — Inovação digital vs. privacidade e anúncios excessivos.
- **Tesla** 🚗⚡ — Futurismo, status e promessas vs. acessibilidade e suporte ao consumidor.
- **Google** 🌍🔍 — Ferramenta essencial vs. invasão de dados e monopólio digital.

Não estou interessada só na polaridade (*positivo/negativo*), mas nos **padrões linguísticos que sustentam essas percepções**.
Espero conseguir aprimorar esse código no futuro para chegar no resultado que imagino.

## 🛠️ **O que o código faz?**

### 📊 Criação do dataset
- Selecionei frases sobre marcas.
- Rotulei cada uma como **positiva, negativa ou neutra**.
- Os verbos e adjetivos foram analisados para entender o tom de cada sentença.

### 🔤 Tokenização com BERT
- O `BertTokenizer` converte os textos para um formato numérico.
- O modelo `bert-base-uncased` gera embeddings contextuais que ajudam na classificação.

### 🏋️ Treinamento do modelo
- Separei os dados em **80% treino e 20% teste**.
- O modelo aprendeu a associar padrões linguísticos a sentimentos.
- Usei `CrossEntropyLoss` e `AdamW` para otimização.

### (ง ͠ಥ_ಥ)ง Criação da API
- Recebe textos via requisição `POST`.
- O modelo processa e devolve a classificação (`positivo`, `neutro` ou `negativo`).
- Erros são tratados de forma clara.

## 💫 **Como rodar isso?**

### 1️⃣ Executar o código
```bash
python nome_do_arquivo.py
```

### 2️⃣ Testar a API
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "A Tesla tem um design incrível!"}'
```

### 🔚 Saída esperada:
```json
{"sentimento": "positivo"}
```

## ⭐ **Por que isso é útil?**
- Empresas podem entender **o impacto emocional de suas estratégias**.
- O comportamento linguístico do consumidor pode indicar padrões de aceitação ou rejeição.
- Ferramenta útil para **monitoramento de redes sociais e feedbacks**.

## 📌 **O que vem depois?**
✅ Expandir o dataset com exemplos reais.  
✅ Ajustar hiperparâmetros para refinar o modelo.  
✅ Explorar emoções mais específicas além de positivo/negativo.  

✉️ **Dúvidas, surtos ou sugestões? Deixe seu recado após o beep.** BEEEEEEP! 

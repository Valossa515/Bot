import spacy
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import json
import time
import random
import threading
import requests

nlp = spacy.load("pt_core_news_lg")

# Carregar perguntas e respostas
def carregar_perguntas_respostas():
    with open('perguntas_respostas.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data["perguntas"], data["respostas"]

perguntas_usuario, respostas = carregar_perguntas_respostas()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(perguntas_usuario)

X = tokenizer.texts_to_sequences(perguntas_usuario)
X = pad_sequences(X, padding='post')

y = np.array([0, 1, 2, 3, 4, 5])  # Classes de respostas

# Definir modelo
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=X.shape[1]))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X, y, epochs=100, batch_size=1, verbose=1, callbacks=[early_stopping])

def responder_pergunta(pergunta):
    pergunta_seq = tokenizer.texts_to_sequences([pergunta])
    pergunta_pad = pad_sequences(pergunta_seq, padding='post', maxlen=X.shape[1])
    predicao = model.predict(pergunta_pad)
    resposta_idx = np.argmax(predicao)
    return respostas[resposta_idx]

def salvar_gostos(gostos):
    try:
        with open('gostos.json', 'r+', encoding='utf-8') as file:
            data = json.load(file)
            data["gostos"].extend(gostos)
            file.seek(0)
            json.dump(data, file, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        with open('gostos.json', 'w', encoding='utf-8') as file:
            json.dump({"gostos": gostos}, file, ensure_ascii=False, indent=4)

def carregar_gostos():
    try:
        with open('gostos.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data["gostos"]
    except FileNotFoundError:
        return []

def perguntar_sobre_gostos(gostos, contador):
    perguntas = [
        "E sobre comida, tem algo que você gosta?",
        "Tem algum filme ou série que você gosta?",
        "Qual seu jogo favorito?",
        "Você tem um hobby?"
    ]
    
    if contador % 5 == 0:
        print(f"Bot: {random.choice(perguntas)}")
        novo_gosto = input("Você: ")
        gostos.append(novo_gosto)
        salvar_gostos([novo_gosto])
        print(f"Bot: Agora eu sei que você gosta de {novo_gosto}!")

def identificar_e_salvar_gosto(frase, gostos):
    if "gosto de" in frase.lower():
        novo_gosto = frase.lower().replace("gosto de", "").strip()
        novo_gosto = novo_gosto.replace("eu", "").strip()
        if novo_gosto and novo_gosto not in gostos:
            gostos.append(novo_gosto)
            salvar_gostos([novo_gosto])
            print(f"Bot: Agora você gosta de {novo_gosto}!")
        elif novo_gosto in gostos:
            print(f"Bot: Você já me disse que gosta de {novo_gosto}.")
    else:
        print("Bot: Não entendi muito bem, pode repetir?")

def listar_gostos(gostos):
    if gostos:
        print(f"Bot: Seus gostos são: {', '.join(gostos)}.")
    else:
        print("Bot: Ainda não sei do que você gosta. Me conte algo!")

def escolher_fonte(termo):
  termo = termo.lower()

  fontes = {
      "wikipedia": ["o que é", "quem foi", "história de", "sobre", "definição de"],
      "google": ["notícia", "últimas novidades", "artigo sobre", "como fazer"],
      "bing": ["onde fica", "mapa de", "previsão do tempo"],
      "openai": ["explicação técnica", "como funciona", "exemplo de código"]
  }

  for fonte, palavras_chave in fontes.items():
    if any(palavra in termo for palavra in palavras_chave):
      return fonte
  return "wikipedia"

def pesquisar_na_web(termo):
  fonte = escolher_fonte(termo)
  if fonte == "wikipedia":
        url = f"https://pt.wikipedia.org/api/rest_v1/page/summary/{termo.replace(' ', '_')}"
  elif fonte == "google":
        url = f"https://www.googleapis.com/customsearch/v1?q={termo}&key=SUA_CHAVE_API&cx=SEU_CX_ID"
  elif fonte == "bing":
        url = f"https://api.bing.microsoft.com/v7.0/search?q={termo}"
  elif fonte == "openai":
        return "Sou um bot básico, ainda não posso acessar a OpenAI, mas posso te ajudar com explicações!"
  
  response = requests.get(url)
  if response.status_code == 200:
    data = response.json()
    if fonte == "wikipedia":
       return data.get("extract", "Não encontrei nada sobre isso.")
    elif fonte == "google":
       return data["items"][0]["snippet"] if "items" in data else "Nada encontrado no Google."
    elif fonte == "bing":
       return data["webPages"]["value"][0]["snippet"] if "webPages" in data else "Nada encontrado no Bing."
  return "Não consegui encontrar informações sobre isso."
  
# Controle de tempo de inatividade
class MonitorInatividade(threading.Thread):
    def __init__(self):
        super().__init__()
        self.ativo = True
        self.ultimo_tempo = time.time()

    def atualizar_tempo(self):
        self.ultimo_tempo = time.time()

    def run(self):
        while self.ativo:
            time.sleep(5)  # Verifica a cada 5 segundos
            if time.time() - self.ultimo_tempo > 10:  # 10 segundos de inatividade
                mensagens = [
                    "Ei, ainda está aí?",
                    "Se precisar de algo, estou por aqui!",
                    "Quer conversar sobre alguma coisa?",
                    "Posso te ajudar com alguma dúvida?"
                ]
                print(f"Bot: {random.choice(mensagens)}")
                self.ultimo_tempo = time.time()  # Evita múltiplas mensagens seguidas

    def parar(self):
        self.ativo = False

def interagir_com_usuario():
    print("Bot: Oi, tudo bem? Como posso ajudar você?")
    time.sleep(0.5)

    gostos = carregar_gostos()
    contador_interacoes = 0

    monitor_inatividade = MonitorInatividade()
    monitor_inatividade.start()

    while True:
        pergunta = input("Você: ").strip().lower()
        monitor_inatividade.atualizar_tempo()  # Atualiza tempo quando o usuário interage

        contador_interacoes += 1

        if pergunta == 'sair':
            print("Bot: Até logo!")
            monitor_inatividade.parar()
            break

        if 'quais são os meus gostos?' in pergunta:
            listar_gostos(gostos)
        elif 'gosta' in pergunta:
            resposta = f"Você gosta de: {', '.join(gostos)}." if gostos else "Ainda não sei do que você gosta."
            print(f"Bot: {resposta}")
        elif 'gosto de' in pergunta:
            identificar_e_salvar_gosto(pergunta, gostos)
        elif "pesquisar sobre" in pergunta or "o que é" in pergunta:
            termo = pergunta.replace("pesquisar sobre", "").replace("o que é", "").strip()
            resposta = pesquisar_na_web(termo)
            print(f"Bot: {resposta}")
        else:
            resposta = responder_pergunta(pergunta)
            print(f"Bot: {resposta}")

        perguntar_sobre_gostos(gostos, contador_interacoes)

# Iniciar interação
interagir_com_usuario()

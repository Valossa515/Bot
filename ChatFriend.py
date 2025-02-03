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

# Carregar modelo NLP para entender frases
nlp = spacy.load("pt_core_news_lg")

# Carregar perguntas e respostas do arquivo JSON
def carregar_perguntas_respostas():
    with open('perguntas_respostas.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data["perguntas"], data["respostas"]

# Carregar perguntas e respostas
perguntas_usuario, respostas = carregar_perguntas_respostas()

# Tokenização e padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(perguntas_usuario)

X = tokenizer.texts_to_sequences(perguntas_usuario)
X = pad_sequences(X, padding='post')

# Respostas em formato numérico
y = np.array([0, 1, 2, 3, 4, 5])  # Cada resposta é uma classe

# Definir modelo
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=X.shape[1]))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))  # Regularização com Dropout
model.add(Dense(6, activation='softmax'))  # 6 classes de respostas

# Compilar modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinamento com early stopping
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Treinar o modelo
model.fit(X, y, epochs=100, batch_size=1, verbose=1, callbacks=[early_stopping])

# Função de resposta do modelo
def responder_pergunta(pergunta):
    pergunta_seq = tokenizer.texts_to_sequences([pergunta])
    pergunta_pad = pad_sequences(pergunta_seq, padding='post', maxlen=X.shape[1])
    
    predicao = model.predict(pergunta_pad)
    resposta_idx = np.argmax(predicao)  # Pegando o índice da maior probabilidade
    return respostas[resposta_idx]

# Função para iniciar uma conversa por conta própria
def iniciar_conversa_propria():
    saudacoes = [
        "Oi, tudo bem? Como posso ajudar você?",
        "Olá! Em que posso ser útil hoje?",
        "Oi! Estou aqui para conversar. O que você gostaria de saber?",
        "Olá! Posso te ajudar com alguma coisa?"
    ]
    print(random.choice(saudacoes))
    time.sleep(1)  # Pausa para dar tempo ao usuário de processar a saudação inicial

# Função para salvar os gostos em um arquivo JSON
def salvar_gostos(gostos):
    try:
        with open('gostos.json', 'r+', encoding='utf-8') as file:
            data = json.load(file)
            data["gostos"].extend(gostos)  # Adiciona novos gostos
            file.seek(0)
            json.dump(data, file, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        with open('gostos.json', 'w', encoding='utf-8') as file:
            json.dump({"gostos": gostos}, file, ensure_ascii=False, indent=4)

# Função para carregar gostos de um arquivo JSON
def carregar_gostos():
    try:
        with open('gostos.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data["gostos"]
    except FileNotFoundError:
        return []

# Função para perguntar sobre gostos caso o usuário não tenha dito ainda
def perguntar_sobre_gostos(gostos):
    if not gostos:  # Se a lista de gostos estiver vazia
        print("Bot: Eu ainda não sei o que você gosta. Me conte, o que você gosta?")
        novo_gosto = input("Você: ")
        gostos.append(novo_gosto)
        salvar_gostos([novo_gosto])
        print(f"Bot: Agora você gosta de {novo_gosto}!")
        
# Função para identificar e salvar o novo gosto
def identificar_e_salvar_gosto(frase, gostos):
    if "gosto de" in frase.lower():  # Verifica se a frase contém "gosto de"
        novo_gosto = frase.lower().replace("gosto de", "").strip()  # Extrai o gosto
        novo_gosto = novo_gosto.replace("eu", "").strip()  # Remover a palavra "eu" se ela existir
        if novo_gosto and novo_gosto not in gostos:  # Evitar adicionar gostos duplicados e verificar se não está vazio
            gostos.append(novo_gosto)
            salvar_gostos([novo_gosto])  # Salva o gosto no arquivo JSON
            print(f"Bot: Agora você gosta de {novo_gosto}!")
        elif novo_gosto in gostos:
            print(f"Bot: Você já disse que gosta de {novo_gosto}.")
        else:
            print("Bot: Não consegui entender. Você pode tentar novamente?")
    else:
        print("Bot: Eu não entendi muito bem. Você pode dizer o que mais gosta?")

# Função para interagir com o usuário
def interagir_com_usuario():
    iniciar_conversa_propria()  # O bot inicia a conversa automaticamente
    
    gostos = carregar_gostos()  # Carregar gostos registrados

    # Perguntar sobre gostos se ainda não foi dito
    perguntar_sobre_gostos(gostos)

    while True:
        pergunta = input("Você: ")
        if pergunta.lower() == 'sair':
            print("Até logo!")
            break

        if 'gosta' in pergunta.lower() or 'gostar' in pergunta.lower():  # Pergunta sobre gostos
            resposta = f"Você gosta de: {', '.join(gostos)}."
            print(f"Bot: {resposta}")
        elif 'gosto de' in pergunta.lower():  # Se o usuário disser "gosto de algo"
            identificar_e_salvar_gosto(pergunta, gostos)
        else:
            resposta = responder_pergunta(pergunta)
            print(f"Bot: {resposta}")

# Iniciar interação
interagir_com_usuario()

import numpy as np
import random
import spacy
import json
import time

# Carregar modelo NLP para entender frases
nlp = spacy.load("pt_core_news_lg")

# Frases do usuário (Estados)
frases_usuario = ["oi", "olá", "bom dia", "boa tarde", "boa noite", 
                  "como você está?", "tudo bem?", "adeus", "tchau", "até mais"]

# Respostas possíveis do chatbot (Ações)
respostas = ["Olá! Como posso ajudar?", "Oi, tudo bem?", "Bom dia!", "Boa tarde!", "Boa noite!", 
             "Estou bem, e você?", "Tudo ótimo!", "Até logo!", "Tchau!", "Nos vemos em breve!"]

# Criando a Q-Table com valores zerados
q_table = np.zeros((len(frases_usuario), len(respostas)))

# Parâmetros de aprendizado por reforço
alpha = 0.1  # Taxa de aprendizado
gamma = 0.6  # Fator de desconto
epsilon = 0.1  # Exploração vs Exploração

# Q-Table para decidir quando iniciar uma conversa
q_table_iniciar = np.zeros((3, 2))  # Estados: [muito_tempo, moderado, recente] | Ações: [não falar, falar]

# Parâmetros para decisão de iniciar conversa
taxa_aprendizado = 0.1
desconto = 0.9
exploracao = 0.2

# Última interação do usuário
ultima_interacao = time.time()

# Banco de dados de preferências (JSON)
PREFERENCIAS_FILE = "preferencias_usuario.json"

def carregar_preferencias():
    try:
        with open(PREFERENCIAS_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def salvar_preferencias(preferencias):
    with open(PREFERENCIAS_FILE, "w") as file:
        json.dump(preferencias, file)

preferencias_usuario = carregar_preferencias()

def processar_texto(texto):
    doc = nlp(texto.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def encontrar_estado(entrada):
    entrada_processada = processar_texto(entrada)
    similaridades = [nlp(entrada_processada).similarity(nlp(processar_texto(frase))) for frase in frases_usuario]
    estado = np.argmax(similaridades)
    return estado, max(similaridades)

def escolher_resposta(estado):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, len(respostas) - 1)
    return np.argmax(q_table[estado])

def treinar_chatbot(episodios=100000):
    for _ in range(episodios):
        estado = random.randint(0, len(frases_usuario) - 1)
        acao = escolher_resposta(estado)
        recompensa = 5 if respostas[acao] in [respostas[estado], respostas[estado - 1]] else -3
        q_table[estado, acao] = (1 - alpha) * q_table[estado, acao] + alpha * (recompensa + gamma * np.max(q_table[estado]))

# Treinar antes de rodar
treinar_chatbot()

def detectar_preferencia(entrada):
    palavras = entrada.lower().split()
    if "gosto de" in entrada or "eu amo" in entrada or "prefiro" in entrada:
        item = entrada.split("de ")[-1] if "de " in entrada else entrada.split("amo ")[-1]
        
        # Carregar as preferências existentes antes de modificar
        preferencias_usuario = carregar_preferencias()

        # Adicionar o novo gosto sem sobrescrever os antigos
        preferencias_usuario[item] = True
        salvar_preferencias(preferencias_usuario)

        return f"Legal! Vou lembrar que você gosta de {item}."
    return None

# 🚀 NOVA FUNÇÃO: Determinar estado baseado no tempo da última interação
def obter_estado_tempo():
    tempo_passado = time.time() - ultima_interacao
    if tempo_passado > 120:
        return 0  # muito_tempo_sem_falar
    elif tempo_passado > 60:
        return 1  # moderado_tempo
    return 2  # recente

# 🚀 NOVA FUNÇÃO: Escolher se o bot deve iniciar uma conversa
def bot_deve_falar():
    estado_tempo = obter_estado_tempo()
    if random.uniform(0, 1) < exploracao:
        return random.choice([0, 1])
    return np.argmax(q_table_iniciar[estado_tempo])

# 🚀 NOVA FUNÇÃO: Aprendizado por reforço para decidir quando falar
def atualizar_q_table_iniciar(recompensa):
    estado_tempo = obter_estado_tempo()
    acao = bot_deve_falar()
    proximo_estado = obter_estado_tempo()
    q_table_iniciar[estado_tempo, acao] = q_table_iniciar[estado_tempo, acao] + taxa_aprendizado * (
        recompensa + desconto * np.max(q_table_iniciar[proximo_estado]) - q_table_iniciar[estado_tempo, acao]
    )

# 🚀 NOVA FUNÇÃO: Bot inicia conversa se achar necessário
def iniciar_conversa():
    global ultima_interacao
    if bot_deve_falar() == 1:
        print("\nChatbot: Oi! Quer conversar?")
        resposta = input("Você: ").strip().lower()
        ultima_interacao = time.time()
        
        if resposta in ["sim", "claro", "vamos conversar", "quero falar"]:
            atualizar_q_table_iniciar(1)  # Recompensa positiva
        elif resposta in ["não", "agora não", "sai", "mais tarde"]:
            atualizar_q_table_iniciar(-1)  # Penalidade
        else:
            atualizar_q_table_iniciar(0)  # Neutro

# 🚀 🚀 Chatbot interativo que agora decide quando falar!
def chatbot():
    global ultima_interacao

    print("Chatbot: Olá! Como posso ajudar? (Digite 'sair' para encerrar)")
    
    while True:
        iniciar_conversa()  # O bot pode tentar iniciar uma conversa
        entrada = input("Você: ").strip().lower()
        
        if entrada == "sair":
            print("Chatbot: Até logo!")
            break
        
        ultima_interacao = time.time()  # Atualiza tempo da última interação
        
        # Verificar se o usuário menciona algo que gosta
        resposta_preferencia = detectar_preferencia(entrada)
        if resposta_preferencia:
            print("Chatbot:", resposta_preferencia)
            continue
        
        # Buscar estado na Q-Table
        estado, confianca = encontrar_estado(entrada)

        if confianca > 0.5:
            melhor_acao = np.argmax(q_table[estado])
            resposta = respostas[melhor_acao]

            # Personalizar resposta se o chatbot souber dos gostos do usuário
            if any(palavra in entrada for palavra in preferencias_usuario):
                resposta += f" Ah, e eu lembro que você gosta de {', '.join(preferencias_usuario.keys())}! 😉"

            print("Chatbot:", resposta)
        else:
            print("Chatbot: Desculpe, não entendi.")

# Iniciar chatbot
chatbot()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Carregar o modelo pré-treinado DialoGPT-large
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Função para rodar o chatbot no terminal
def chat_with_ai():
    print("Você pode começar a conversar comigo! Escreva 'sair' para encerrar a conversa.")
    chat_history_ids = None  # Histórico da conversa
    max_history_length = 3  # Limitar o histórico a 3 interações anteriores

    while True:
        user_input = input("Você: ")

        if user_input.lower() in ["sair", "adeus", "tchau"]:
            print("Chatbot: Até logo!")
            break

        # Codificar a entrada do usuário e incluir o histórico da conversa
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

        # Limitar o histórico da conversa para evitar confusão
        if chat_history_ids is not None:
            chat_history_ids = chat_history_ids[:, -max_history_length:]

        # Criar a `attention_mask` para a entrada do usuário
        attention_mask = torch.ones(new_input_ids.shape, dtype=torch.long)

        # Gerar uma resposta com o histórico da conversa
        chat_history_ids = model.generate(
            torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids,
            attention_mask=torch.cat([torch.ones_like(chat_history_ids), attention_mask], dim=-1) if chat_history_ids is not None else attention_mask,
            max_length=500,  # Limite máximo do comprimento de resposta
            pad_token_id=tokenizer.eos_token_id,  # Preencher tokens
            no_repeat_ngram_size=3,  # Evitar repetição
            do_sample=True,  # Amostrar a partir da distribuição de probabilidades
            top_k=50,  # Número de tokens a serem considerados
            top_p=0.9,  # Filtro nucleus (menor valor para respostas mais previsíveis)
            temperature=0.6,  # Controla a "criatividade" das respostas (mais baixo para maior coerência)
        )

        # Decodificar a resposta gerada
        response = tokenizer.decode(chat_history_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Chatbot: {response}")

# Rodar o chatbot
if __name__ == "__main__":
    chat_with_ai()

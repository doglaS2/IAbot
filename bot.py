from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Carregamento do modelo pré-treinado DialoGPT-large
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

def chat_with_ai():
    """
    Executa um chatbot interativo no terminal utilizando o modelo DialoGPT.

    O chatbot mantém um histórico limitado da conversa para gerar respostas mais contextuais.
    A conversa é encerrada quando o usuário digita uma das palavras-chave de saída.

    Entradas do usuário:
        - Texto livre digitado no terminal.
        - Palavras-chave como 'sair', 'adeus' ou 'tchau' encerram a conversa.

    Parâmetros internos:
        - `chat_history_ids` (torch.Tensor or None): histórico de interações anteriores, usado como
          entrada adicional ao modelo para gerar respostas contextualizadas.
        - `max_history_length` (int): número máximo de interações anteriores que o modelo irá considerar.
        - `attention_mask` (torch.Tensor): máscara de atenção indicando quais tokens devem ser considerados.

    Geração de resposta:
        - A resposta é gerada com amostragem (`do_sample=True`) para permitir variedade.
        - Técnicas de controle como `top_k`, `top_p`, `no_repeat_ngram_size` e `temperature`
          são usadas para refinar a geração de texto.

    Retorno:
        - A função imprime as respostas do chatbot diretamente no terminal.
        - Não há valor de retorno explícito.
    """
    print("Você pode começar a conversar comigo! Escreva 'sair' para encerrar a conversa.")
    chat_history_ids = None
    max_history_length = 3

    while True:
        user_input = input("Você: ")

        if user_input.lower() in ["sair", "adeus", "tchau"]:
            print("Chatbot: Até logo!")
            break

        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

        if chat_history_ids is not None:
            chat_history_ids = chat_history_ids[:, -max_history_length:]

        attention_mask = torch.ones(new_input_ids.shape, dtype=torch.long)

        chat_history_ids = model.generate(
            torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids,
            attention_mask=torch.cat([torch.ones_like(chat_history_ids), attention_mask], dim=-1) if chat_history_ids is not None else attention_mask,
            max_length=500,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.6,
        )

        response = tokenizer.decode(chat_history_ids[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat_with_ai()
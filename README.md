# IAbot

IAbot é um chatbot interativo que utiliza o modelo DialoGPT-large da Microsoft para gerar respostas em uma conversa de terminal. Ele utiliza a biblioteca [Transformers](https://github.com/huggingface/transformers) e o [PyTorch](https://pytorch.org/) para implementar a interação.

## Como Funciona

- **Carregamento do Modelo:**  
  O script [bot.py](insira_aqui_seu_diretorio/bot.py) importa e carrega o modelo DialoGPT-large e seu respectivo tokenizer.

- **Histórico de Conversa:**  
  A função `chat_with_ai` mantém um histórico das últimas três interações para preservar algum contexto na conversa.

- **Processamento de Entrada:**  
  Quando o usuário digita uma mensagem, o texto é codificado em tokens utilizando o tokenizer, adicionando um token de finalização de sequência (EOS) automaticamente.

- **Geração de Resposta:**  
  O modelo gera uma resposta utilizando técnicas de amostragem (`do_sample=True`) com parâmetros como `top_k`, `top_p`, `no_repeat_ngram_size` e `temperature` para controlar a diversidade e evitar repetições. O histórico anterior (se existir) é concatenado com a nova entrada para fornecer contexto à geração.

- **Saída da Conversa:**  
  A resposta gerada é decodificada a partir dos tokens para texto e impressa no terminal para o usuário. A conversa continua até que o usuário digite uma palavra de saída (por exemplo, "sair", "adeus" ou "tchau").

## Requisitos

- Python 3.x
- [Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)

## Como Executar

Para iniciar o chatbot, execute o script `bot.py` a partir do terminal:

```sh
python [bot.py](http://_vscodecontentref_/1)# IAbot
 

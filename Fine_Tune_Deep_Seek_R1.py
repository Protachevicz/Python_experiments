import os
import torch

# Libera a memória da GPU antes de iniciar o processo de treinamento
# Isso pode ajudar a evitar problemas de alocação de memória
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Importação das bibliotecas necessárias para o treinamento do modelo
import transformers
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model

def load_data():
    """
    Carrega o dataset para fine-tuning.
    Neste caso, estamos utilizando o conjunto de dados 'medical-o1-reasoning-SFT',
    que é útil para tarefas de modelagem de linguagem na área médica.
    """
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
    
    # Divide o dataset em conjunto de treino (90%) e teste (10%)
    dataset = dataset["train"].train_test_split(test_size=0.1)
    return dataset

def preprocess_function(examples, tokenizer):
    """
    Pré-processa os dados para o modelo.
    Tokeniza as perguntas presentes no dataset para que possam ser utilizadas no modelo.
    
    Parâmetros:
        examples (dict): Um dicionário contendo os exemplos do dataset.
        tokenizer (AutoTokenizer): Tokenizador correspondente ao modelo.
    
    Retorna:
        Um dicionário com os textos tokenizados.
    """
    return tokenizer(examples["Question"], truncation=True, padding="max_length", max_length=512)

def main():
    # Nome do modelo base a ser utilizado
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Modelo alternativo
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Modelo atual
    output_dir = "./fine_tuned_deepseek"  # Diretório onde o modelo ajustado será salvo
    
    # Configuração para quantização 4-bit (QLoRA)
    # Permite treinar modelos grandes com menor uso de memória GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Ativa a quantização em 4 bits
        bnb_4bit_compute_dtype=torch.float16,  # Usa float16 para cálculos durante o treinamento
        bnb_4bit_use_double_quant=True,  # Ativa quantização dupla para maior eficiência
        bnb_4bit_quant_type="nf4"  # Tipo de quantização específica para eficiência computacional
    )
    
    # Carregamento do tokenizador e modelo base
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    
    # Desativa o uso de cache do modelo durante o treinamento para evitar problemas
    model.config.use_cache = False  
    
    # Configuração do LoRA (Low-Rank Adaptation)
    # Adapta o modelo utilizando um número reduzido de parâmetros treináveis
    lora_config = LoraConfig(
        r=8,  # Rank do LoRA (reduzido para eficiência)
        lora_alpha=16,  # Parâmetro de escala para aprendizado
        lora_dropout=0.1,  # Dropout para regularização
        bias="none",  # Não ajusta bias do modelo
        task_type="CAUSAL_LM"  # Define a tarefa como modelagem de linguagem causal
    )
    
    # Aplica LoRA ao modelo
    model = get_peft_model(model, lora_config)
    
    # Carregar e pré-processar dataset
    dataset = load_data()
    tokenized_datasets = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    
    # Configuração do treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,  # Diretório de saída para salvar checkpoints
        per_device_train_batch_size=1,  # Define o tamanho do batch para treinamento
        per_device_eval_batch_size=1,  # Define o tamanho do batch para avaliação
        gradient_accumulation_steps=4,  # Acumula gradientes por 4 passos para treinar modelos maiores
        evaluation_strategy="epoch",  # Avalia o modelo ao final de cada época
        save_strategy="epoch",  # Salva o modelo ao final de cada época
        learning_rate=2e-4,  # Taxa de aprendizado para ajuste fino do modelo
        num_train_epochs=3,  # Número total de épocas de treinamento
        fp16=True,  # Ativa treinamento em float16 para melhor eficiência
        logging_dir="./logs",  # Diretório para armazenar logs do treinamento
        report_to="none"  # Evita que os logs sejam enviados para plataformas externas
    )
    
    # Configuração do Trainer
    trainer = Trainer(
        model=model,  # Modelo a ser treinado
        args=training_args,  # Argumentos de configuração do treinamento
        train_dataset=tokenized_datasets["train"],  # Conjunto de treinamento
        eval_dataset=tokenized_datasets["test"],  # Conjunto de avaliação
        tokenizer=tokenizer,  # Tokenizador correspondente ao modelo
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)  # Define o collator de dados para modelagem de linguagem
    )
    
    # Inicia o processo de fine-tuning
    trainer.train()
    
    # Salva o modelo e o tokenizador após o treinamento
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Fine-tuning concluído!")

# Executa o script apenas se for chamado diretamente
if __name__ == "__main__":
    main()


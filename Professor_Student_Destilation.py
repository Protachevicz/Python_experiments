import torch  # Biblioteca principal para computação com tensores e redes neurais
import torch.nn as nn  # Módulo para construção de redes neurais
import torch.optim as optim  # Otimizadores para treinamento
import pandas as pd  # Biblioteca para manipulação de dados tabulares
import random  # Módulo para geração de números aleatórios
from torch.utils.data import Dataset, DataLoader  # Classes para manipulação de dados em batches
from transformers import AutoTokenizer  # Tokenizador de modelos pré-treinados do Hugging Face

# Configuração do dispositivo para uso de GPU, se disponível, senão usa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definição de hiperparâmetros para o treinamento do modelo aluno
epochs_distillation = 100000  # Número de épocas para treinamento
batch_size = 1000  # Tamanho do lote para processamento dos dados

# Carregamento do tokenizador BERT-base (uncased: sem diferenciação de maiúsculas/minúsculas)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Dicionário contendo os caminhos dos datasets de treino, validação e teste
splits = {
    'train': "hf://datasets/Goastro/mlx-grpo-dataset/data/train-00000-of-00001.parquet",
    'valid': "hf://datasets/Goastro/mlx-grpo-dataset/data/valid-00000-of-00001.parquet",
    'test': "hf://datasets/Goastro/mlx-grpo-dataset/data/test-00000-of-00001.parquet"
}

# Função para carregar e pré-processar dados em formato Parquet
def load_and_preprocess_data(filepath):
    """Carrega um dataset Parquet e aplica pré-processamento de texto."""
    df = pd.read_parquet(filepath)  # Carrega o arquivo Parquet em um DataFrame
####Como a recompensa de acurácia é calculada. Resposta: Nao Apliquei o RL.
####omo a recompensa de formatação é verificada. REsposta: Nao apliquei o RL. Para contornar isso apenas pedi para o programa ignorar os espaços vazios
    df['prompt'] = df['prompt'].str.lower().str.strip()  # Converte para minúsculas e remove espaços extras
    df['answer'] = df['answer'].str.lower().str.strip()
    return df  # Retorna o dataframe processado

# Carregar os conjuntos de treino, validação e teste
dtrain = load_and_preprocess_data(splits["train"])
dvalid = load_and_preprocess_data(splits["valid"])
dtest = load_and_preprocess_data(splits["test"])

# Função para converter texto em tensores numéricos usando BERT
# text_to_tensor(text) converte um texto bruto (string) em um tensor numérico
#  utilizando a tokenização do modelo pré-treinado BERT (Bidirectional Encoder Representations from Transformers). 
# A conversão é essencial para que o texto possa ser processado por um modelo de rede neural.
def text_to_tensor(text):
    """Converte texto para tensores numéricos usando tokenização BERT."""
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=50, return_tensors="pt")
    return tokens.input_ids.squeeze(0).to(torch.float32) # Converte para float32 para compatibilidade com o modelo

# Definição da classe de Dataset personalizada
class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe  # Armazena o dataframe para acesso posterior
    
    def __len__(self):
        return len(self.data)  # Retorna o número total de exemplos no dataset
    
    def __getitem__(self, idx):
        # Obtém o texto do prompt e da resposta correspondente
        prompt = self.data.iloc[idx]['prompt']
        answer = self.data.iloc[idx]['answer']
        
        # Converte os textos em tensores numéricos compatíveis com a rede neural
        prompt_tensor = text_to_tensor(prompt)
        answer_tensor = text_to_tensor(answer)
        
        return prompt_tensor, answer_tensor, prompt, answer  # Retorna tensores e textos brutos para análise

# Criando instâncias dos datasets
train_dataset = TextDataset(dtrain)
val_dataset = TextDataset(dvalid)
test_dataset = TextDataset(dtest)

# Criando DataLoaders para alimentar o modelo em batches
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Embaralha para evitar viés
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definição de um modelo neural simples usando camadas totalmente conectadas
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Primeira camada totalmente conectada
        self.relu = nn.ReLU()  # Função de ativação ReLU para introduzir não-linearidade
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Segunda camada totalmente conectada
    
    def forward(self, x):
        x = self.fc1(x)  # Passagem pela primeira camada
        x = self.relu(x)  # Aplicação da ativação ReLU
        x = self.fc2(x)  # Passagem pela segunda camada e saída
        return x  # Retorna a saída do modelo

# Definição da arquitetura dos modelos Professor e Aluno
input_dim = 50  # Dimensão de entrada (tamanho do embedding)
hidden_dim = 1500  # Número de neurônios na camada oculta do modelo Professor
output_dim = 50  # Dimensão da saída, igual à entrada (autoencoder)

# Criando os modelos Professor e Aluno
professor_model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)
aluno_model = SimpleNN(input_dim, 5000, output_dim).to(device)  # Modelo Aluno com mais neurônios na camada oculta

# Definição do otimizador e função de perda
optimizer_aluno = optim.Adam(aluno_model.parameters(), lr=0.00001)  # Otimizador Adam com taxa de aprendizado pequena
loss_fn = nn.MSELoss()  # Função de perda MSE (Mean Squared Error) para minimizar erro quadrático médio

# Loop de treinamento do modelo aluno usando destilação do modelo professor
for epoch in range(epochs_distillation):
    aluno_model.train()  # Coloca o modelo em modo de treinamento
    total_loss = 0  # Acumulador de perda
    
    for prompts_tensor, answers_tensor, _, _ in train_loader:
        prompts_tensor = prompts_tensor.to(device, dtype=torch.float32)  # Move tensores para GPU/CPU conforme necessário

        with torch.no_grad():  # O modelo professor não atualiza os pesos
            professor_outputs = professor_model(prompts_tensor)  # Saída do professor como rótulo do aluno

        aluno_outputs = aluno_model(prompts_tensor)  # Modelo aluno gera previsões
        loss = loss_fn(aluno_outputs, professor_outputs)  # Calcula a perda comparando com a saída do professor
        
        optimizer_aluno.zero_grad()  # Zera os gradientes acumulados
        loss.backward()  # Retropropagação do erro
        optimizer_aluno.step()  # Atualização dos pesos do modelo aluno
        
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)  # Calcula a perda média da época

    # Validação a cada 30 épocas para monitoramento
    if epoch % 1000 == 0 or epoch == epochs_distillation - 1:
        aluno_model.eval()  # Coloca o modelo aluno em modo de avaliação
        val_loss = 0
        with torch.no_grad():  # Desativa gradientes
            for prompts_tensor, answers_tensor, _, _ in val_loader:
                prompts_tensor = prompts_tensor.to(device, dtype=torch.float32)
                professor_outputs = professor_model(prompts_tensor)
                aluno_outputs = aluno_model(prompts_tensor)
                loss = loss_fn(aluno_outputs, professor_outputs)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs_distillation} | Train Loss: {avg_train_loss:.4f}")


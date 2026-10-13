import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Configurações gerais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())  # Verifica se GPU está disponível

epochs_distillation = 2000  # Número de épocas para destilação
batch_size = 100  # Tamanho do lote

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Definição dos caminhos dos conjuntos de dados
splits = {
    'train': "hf://datasets/Goastro/mlx-grpo-dataset/data/train-00000-of-00001.parquet",
    'valid': "hf://datasets/Goastro/mlx-grpo-dataset/data/valid-00000-of-00001.parquet",
    'test': "hf://datasets/Goastro/mlx-grpo-dataset/data/test-00000-of-00001.parquet"
}

# Função para carregar os datasets
def load_data(split):
    return pd.read_parquet(splits[split])

# Carregando os dados
dtrain = load_data("train")
dvalid = load_data("valid")
dtest = load_data("test")

# Função de pré-processamento de texto
def preprocess_text(text):
    """Remove espaços extras e converte para letras minúsculas."""
    return " ".join(text.lower().strip().split())

# Aplicando o pré-processamento nos conjuntos de dados
dtrain['prompt'] = dtrain['prompt'].apply(preprocess_text)
dtrain['answer'] = dtrain['answer'].apply(preprocess_text)
dvalid['prompt'] = dvalid['prompt'].apply(preprocess_text)
dvalid['answer'] = dvalid['answer'].apply(preprocess_text)
dtest['prompt'] = dtest['prompt'].apply(preprocess_text)
dtest['answer'] = dtest['answer'].apply(preprocess_text)

# Função para converter texto em tensores numéricos
def text_to_tensor(text):
    """Converte texto para embeddings numéricos usando tokenização BERT."""
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=50, return_tensors="pt")
    return tokens.input_ids.squeeze(0)

# Classe personalizada para manipulação dos dados
class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt = text_to_tensor(self.data.iloc[idx]['prompt'])
        answer = text_to_tensor(self.data.iloc[idx]['answer'])
        return prompt, answer

# Criando os conjuntos de dados e dataloaders
dataset_train = TextDataset(dtrain)
dataset_valid = TextDataset(dvalid)
dataset_test = TextDataset(dtest)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Definição da rede neural
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instanciando o modelo professor e o modelo aluno
input_dim = 50  # Dimensão da entrada (número de tokens)
hidden_dim = 500  # Camada oculta
output_dim = 50  # Dimensão da saída (número de tokens)

professor_model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)
aluno_model = SimpleNN(input_dim, hidden_dim // 2, output_dim).to(device)

# Definição do otimizador e da função de perda
optimizer_aluno = optim.Adam(aluno_model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

# Treinamento com destilação
def train_distillation():
    """Treina o modelo aluno imitando a saída do modelo professor."""
    for epoch in range(epochs_distillation):
        total_loss = 0
        for prompts, _ in dataloader_train:
            prompts = prompts.to(device, dtype=torch.float32)
            
            with torch.no_grad():
                professor_outputs = professor_model(prompts)
            
            aluno_outputs = aluno_model(prompts)
            loss = loss_fn(aluno_outputs, professor_outputs)
            
            optimizer_aluno.zero_grad()
            loss.backward()
            optimizer_aluno.step()
            
            total_loss += loss.item()
        
        print(f"Distillation Epoch {epoch+1}/{epochs_distillation} | Distillation Loss: {total_loss / len(dataloader_train):.4f}")
    
    print("Destilação concluída.")

# Validação do modelo
def validate_model():
    """Avalia o modelo aluno no conjunto de validação."""
    aluno_model.eval()
    total_loss = 0
    with torch.no_grad():
        for prompts, _ in dataloader_valid:
            prompts = prompts.to(device, dtype=torch.float32)
            professor_outputs = professor_model(prompts)
            aluno_outputs = aluno_model(prompts)
            loss = loss_fn(aluno_outputs, professor_outputs)
            total_loss += loss.item()
    
    print(f"Validation Loss: {total_loss / len(dataloader_valid):.4f}")
    aluno_model.train()

# Teste do modelo
def test_model():
    """Testa o modelo aluno no conjunto de teste."""
    aluno_model.eval()
    total_loss = 0
    with torch.no_grad():
        for prompts, _ in dataloader_test:
            prompts = prompts.to(device, dtype=torch.float32)
            professor_outputs = professor_model(prompts)
            aluno_outputs = aluno_model(prompts)
            loss = loss_fn(aluno_outputs, professor_outputs)
            total_loss += loss.item()
    
    print(f"Test Loss: {total_loss / len(dataloader_test):.4f}")
    aluno_model.train()

# Executando treinamento, validação e teste
train_distillation()
validate_model()
test_model()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

###
import sys
import os

# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
from util.parameters import INPUT_SIZE, OUTPUT_SIZE, LR, BATCH_SIZE
from util.parameters import NUM_EPOCHS, FILE_PATH, PYTORCH_MODEL_FILE
from util.parameters import RANDOM_STATE, TEST_SIZE, RESULTS_COLUMN
from formatation.input_formatation import load_data, separate_features_labels
from studies.scikit import split_train_test


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.float),
                torch.tensor(self.labels[idx], dtype=torch.long))


def get_data_loaders(X_train, X_test,
                     y_train, y_test,
                     batch_size=32):
    """
    Formatar os dados para o Pytorch
    """
    # Definir transformações de pré-processamento
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Converter para tensores do PyTorch
    train_dataset = CustomDataset(X_train, y_train, transform=preprocess)
    test_dataset = CustomDataset(X_test, y_test, transform=preprocess)

    # Definir os dataloaders
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size)
    return train_loader, test_loader


class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size, lr=0.001):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.running_loss = 0.0
        self.accuracy = 0.0

    def forward(self, x):
        x = self.fc(x)
        return x

    def train_model(self, train_loader):
        '''
        Treinar o modelo usando o conjunto de teste
        e calcular a running loss
        '''
        self.train()
        for inputs, labels in train_loader:
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, labels-1)
            loss.backward()
            self.optimizer.step()
            self.running_loss += loss.item()
        self.running_loss /= len(train_loader)
    
    def test(self, test_loader):
        '''
        Testar o modelo usando o conjunto de teste
        e retornar a acurácia
        '''
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.accuracy = correct/total
    
    def save(self, path=PYTORCH_MODEL_FILE):
        """
        salvar os pesos do modelo em path
        """
        torch.save(self.state_dict(), path)

    def load(self, path=PYTORCH_MODEL_FILE):
        """
        carrega os pesos e muda para o estado de avaliação
        """
        self.load_state_dict(torch.load(path))
        self.eval()


def main():
    # Passo 1: Carregar os dados do CSV
    data = load_data(FILE_PATH)
    
    # Passo 2: Separar features (X) dos labels (Y)
    X, y = separate_features_labels(data, RESULTS_COLUMN)
    
    # Passo 3: Dividir em conjuntos de treino e teste
    (X_train, X_test,
     y_train, y_test) = split_train_test(X, y, TEST_SIZE, RANDOM_STATE)
    # formatar os dados para o PyTorch
    train_loader, test_loader = get_data_loaders(X_train, X_test,
                                                 y_train, y_test,
                                                 BATCH_SIZE)

    # Passo 4: Inicializar o modelo de Classificação
    global model

    best_acc = -1
    for epoch in range(NUM_EPOCHS):
        # Passo 5: Treinar o modelo
        model.train_model(train_loader)

        # Passo 6: Testar o modelo usando o conjunto de teste
        model.test(test_loader)

        # Passo 7 Salvar o melhor modelo treinado
        if model.accuracy > best_acc:
            best_acc = model.accuracy
            model.save()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"Loss: {model.running_loss}")
            print(f"Accuracy: {(model.accuracy)*100:.2f}%\n")

model = SimpleModel(INPUT_SIZE, OUTPUT_SIZE, LR)
# Chamando a função principal para treinar o modelo
if __name__ == "__main__":
    print("\n--------------------\nTraining the pytorch model...")
    main()
    print("\n--------------------\nModel trained successfully!")
else:
    # Carregar modelo treinado
    model.load(PYTORCH_MODEL_FILE)

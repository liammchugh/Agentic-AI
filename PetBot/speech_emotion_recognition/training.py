import torch
import torch.nn as nn
import torch.optim as optim
from SER_architecture import EmotionClassifier, SERModel
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Dataset class for loading audio and labels
class SERDataset(Dataset):
    def __init__(self, audio_files, labels, processor):
        self.audio_files = audio_files
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.audio_files[idx])
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        inputs = self.processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
        label = self.labels[idx]
        return inputs.squeeze(0), torch.tensor(label, dtype=torch.long)



# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        val_acc = validate_model(model, val_loader, criterion)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Acc: {train_acc}, Val Acc: {val_acc}')

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_ser_model.pth')

    print(f'Best Validation Accuracy: {best_val_acc}')

# Validation function
def validate_model(model, val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512)
    batch_size = trial.suggest_int('batch_size', 8, 32)

    # Create DataLoaders
    train_loader = DataLoader(SERDataset(train_files, train_labels, processor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SERDataset(val_files, val_labels, processor), batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    input_dim = 1024  # Feature size from XLSR
    num_classes = 4  # Assuming 4 emotion classes (adjust based on dataset)
    classifier = EmotionClassifier(input_dim, hidden_dim, num_classes)
    ser_model = SERModel(xlsr_model, classifier, layer_to_extract=12)
    
    optimizer = optim.Adam(ser_model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train model
    train_model(ser_model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

    # Evaluate validation accuracy
    val_acc = validate_model(ser_model, val_loader, criterion)

    # Optuna tries to minimize by default, so return -val_acc for maximizing validation accuracy
    return -val_acc
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.fmin import space_eval

# Define search space
space = {
    'lr': hp.loguniform('lr', -5, -3),  # Learning rate (log scale)
    'hidden_dim': hp.quniform('hidden_dim', 128, 512, 1),
    'batch_size': hp.quniform('batch_size', 8, 32, 1)
}

# Objective function
def objective(params):
    lr = params['lr']
    hidden_dim = int(params['hidden_dim'])
    batch_size = int(params['batch_size'])

    # Create DataLoaders
    train_loader = DataLoader(SERDataset(train_files, train_labels, processor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SERDataset(val_files, val_labels, processor), batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = 1024
    num_classes = 4
    classifier = EmotionClassifier(input_dim, hidden_dim, num_classes)
    ser_model = SERModel(xlsr_model, classifier, layer_to_extract=12)

    # Optimizer and criterion
    optimizer = optim.Adam(ser_model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train model
    train_model(ser_model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

    # Validate model
    val_acc = validate_model(ser_model, val_loader, criterion)

    return -val_acc

# Run hyperparameter optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

# Show the best parameters
print(f"Best Hyperparameters: {space_eval(space, best)}")


torch.save({
    'model_state_dict': ser_model.state_dict(),
    'hyperparameters': best_params
}, 'best_ser_model_and_params.pth')

print("Model and hyperparameters saved successfully!")

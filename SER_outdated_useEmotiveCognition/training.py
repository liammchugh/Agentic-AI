import torch
import torch.nn as nn
import torch.optim as optim
from SER_architecture import MLP_EmotionClassifier, SERModel
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils.optimization import WarmupScheduler

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            # Accumulate training loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        val_acc = validate_model(model, val_loader, criterion)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_ser_model.pth')

    print(f'Best Validation Accuracy: {best_val_acc:.4f}')

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

# Objective function for Optuna Hyperparameter Optimization
def objective(trial, train_loader, val_loader):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512)
    batch_size = trial.suggest_int('batch_size', 8, 32)

    # Initialize model, optimizer, and loss function
    input_dim = 1024  # Feature size from XLSR
    num_classes = 4  # Assuming 4 emotion classes (adjust based on dataset)

    classifier = MLP_EmotionClassifier(input_dim, hidden_dim, num_classes)
    # Load XLSR model and processor
    from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor
    model_name = "facebook/wav2vec2-xlsr-53"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    xlsr_model = Wav2Vec2Model.from_pretrained(model_name)
    # Freeze XLSR model parameters (no fine-tuning)
    for param in xlsr_model.parameters():
        param.requires_grad = False    

    ser_model = SERModel(xlsr_model, classifier, layer_to_extract=None)
    # currently running complete model
    optimizer = optim.Adam(ser_model.classifier.parameters(), lr=lr)
    warmup_steps = 50
    total_steps = len(train_loader) * 10  # 10 epochs
    scheduler = WarmupScheduler(optimizer, warmup_steps, total_steps, base_lr=lr*0.01, peak_lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train model
    train_model(ser_model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

    # Evaluate validation accuracy
    val_acc = validate_model(ser_model, val_loader, criterion)

    # Optuna tries to minimize by default, so return -val_acc for maximizing validation accuracy
    return -val_acc


if __name__ == "__main__":

    # Create DataLoaders
    train_loader = DataLoader(SERDataset(train_files, train_labels, processor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SERDataset(val_files, val_labels, processor), batch_size=batch_size, shuffle=False)

    from hyperopt import fmin, tpe, hp, Trials
    from hyperopt.fmin import space_eval

    # Define search space
    space = {
        'lr': hp.loguniform('lr', -5, -3),  # Learning rate (log scale)
        'warmup_steps': hp.quniform('warmup_steps', 5, 100, 5),  # Warmup steps
        'hidden_dim': hp.quniform('hidden_dim', 128, 512, 1),
        'batch_size': hp.quniform('batch_size', 8, 32, 1)
    }

    # Run hyperparameter optimization
    trials = Trials()
    best = fmin(fn=objective(train_loader, val_loader), space=space, algo=tpe.suggest, max_evals=20, trials=trials)

    # Show the best parameters
    print(f"Best Hyperparameters: {space_eval(space, best)}")

    torch.save({
        'model_state_dict': ser_model.state_dict(),
        'hyperparameters': best_params
    }, 'best_ser_model_and_params.pth')

    print("Model and hyperparameters saved successfully!")

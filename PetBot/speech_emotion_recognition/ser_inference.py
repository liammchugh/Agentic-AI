import torch
from torch.utils.data import DataLoader

# Load saved model and hyperparameters
checkpoint = torch.load('best_ser_model_and_params.pth')
saved_state_dict = checkpoint['model_state_dict']
saved_hyperparameters = checkpoint['hyperparameters']

# Rebuild the model with the saved hyperparameters
hidden_dim = saved_hyperparameters['hidden_dim']  # Assuming hidden_dim was one of the saved hyperparameters
batch_size = saved_hyperparameters['batch_size']  # Assuming batch_size was one of the saved hyperparameters
lr = saved_hyperparameters['lr']  # Assuming lr was one of the saved hyperparameters

# Define the classifier and model architecture using the saved hyperparameters
input_dim = 1024  # Same as before, depending on your feature extractor
num_classes = 4   # Number of emotion classes
classifier = EmotionClassifier(input_dim, hidden_dim, num_classes)
ser_model = SERModel(xlsr_model, classifier, layer_to_extract=12)  # XLSR model is pre-trained, layer 12 used

# Load the saved model weights
ser_model.load_state_dict(saved_state_dict)

# Set the model to evaluation mode for inference
ser_model.eval()

print("Model loaded successfully for inference!")

# Now you can perform inference using the loaded model
# Assuming we have a test dataset and DataLoader

def predict_emotion(filepath):
    # Load and process the audio
    waveform = load_audio(filepath)  # Assuming load_audio function from earlier
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values

    # Get emotion prediction
    with torch.no_grad():
        logits = ser_model(input_values)
        predicted_emotion = torch.argmax(logits, dim=-1)
    
    return predicted_emotion.item()

# Example inference
audio_path = "path_to_audio_file.wav"
predicted_emotion = predict_emotion(audio_path)
emotions = ["Neutral", "Happy", "Sad", "Angry"]  # Assuming 4 emotion classes
print(f"Predicted Emotion: {emotions[predicted_emotion]}")

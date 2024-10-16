import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio

# Load XLSR model and processor
model_name = "facebook/wav2vec2-xlsr-53"
processor = Wav2Vec2Processor.from_pretrained(model_name)
xlsr_model = Wav2Vec2Model.from_pretrained(model_name)

# Freeze XLSR model parameters (no fine-tuning)
for param in xlsr_model.parameters():
    param.requires_grad = False

# Classifier head for emotion prediction
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load and preprocess the input audio
def load_audio(filepath):
    waveform, sample_rate = torchaudio.load(filepath)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform

# Define model architecture
class SERModel(nn.Module):
    def __init__(self, xlsr_model, classifier, layer_to_extract):
        super(SERModel, self).__init__()
        self.xlsr_model = xlsr_model
        self.classifier = classifier
        self.layer_to_extract = layer_to_extract  # Extract features from this layer

    def forward(self, input_values):
        # Extract features from XLSR
        with torch.no_grad():
            outputs = self.xlsr_model(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            xlsr_features = hidden_states[self.layer_to_extract]  # Choose a middle layer
            xlsr_features = xlsr_features.mean(dim=1)  # Average over time

        # Pass features through classifier
        logits = self.classifier(xlsr_features)
        return logits

if __name__ == "__main__":
    # Hyperparameters
    input_dim = 1024  # Feature size from XLSR model
    hidden_dim = 128  # Number of hidden units in classifier
    num_classes = 4   # For example, IEMOCAP dataset has 4 emotion classes

    # Create classifier and full SER model
    classifier = EmotionClassifier(input_dim, hidden_dim, num_classes)
    ser_model = SERModel(xlsr_model, classifier, layer_to_extract=12)  # Layer 12 is often optimal

    # Example usage
    def predict_emotion(filepath):
        # Load and process the audio
        waveform = load_audio(filepath)
        input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values

        # Get emotion prediction
        ser_model.eval()
        with torch.no_grad():
            logits = ser_model(input_values)
            predicted_emotion = torch.argmax(logits, dim=-1)
        return predicted_emotion.item()

    # Test the model with an example audio file
    audio_path = "path_to_audio_file.wav"
    emotion_prediction = predict_emotion(audio_path)
    print(f"Predicted Emotion: {emotion_prediction}")

    # Example emotions corresponding to indices (this depends on your dataset)
    emotions = ["Neutral", "Happy", "Sad", "Angry"]
    print(f"Predicted Emotion: {emotions[emotion_prediction]}")

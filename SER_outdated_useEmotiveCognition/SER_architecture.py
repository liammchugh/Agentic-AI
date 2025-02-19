import torch
import torch.nn as nn

# Classifier head for emotion prediction
class MLP_EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP_EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Complete model architecture
class SERModel(nn.Module):
    def __init__(self, xlsr_model, classifier, layer_to_extract=None):
        super(SERModel, self).__init__()
        self.xlsr_model = xlsr_model
        self.classifier = classifier
        if layer_to_extract is not None:
            self.layer_to_extract = layer_to_extract  # Extract features from this layer

    def forward(self, input_values):
        # Extract feature from spec. layer after running XLSR
        with torch.no_grad():
            outputs = self.xlsr_model(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Extract features from specified layer; needs further research
            # https://huggingface.co/facebook/wav2vec2-large-xlsr-53
            if self.layer_to_extract is not None:
                xlsr_features = hidden_states[self.layer_to_extract] 
                outputs = xlsr_features.mean(dim=1)

        # Pass features through classifier
        logits = self.classifier(outputs)

        return logits

if __name__ == "__main__":
    # Hyperparameters
    input_dim = 1024  # Feature size from XLSR model
    hidden_dim = 128  # Number of hidden units in classifier
    num_classes = 4   # For example, IEMOCAP dataset has 4 emotion classes
    # Load model directly
    from transformers import AutoProcessor, AutoModelForPreTraining

    # processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    xlsr_model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")

    # from transformers import Wav2Vec2ForCTC
    # xlsr_model = Wav2Vec2ForCTC.from_pretrained(
    #     "facebook/wav2vec2-xls-r-300m", 
    #     attention_dropout=0.0,
    #     hidden_dropout=0.0,
    #     feat_proj_dropout=0.0,
    #     mask_time_prob=0.05,
    #     layerdrop=0.0,
    #     ctc_loss_reduction="mean", 
    #     pad_token_id=processor.tokenizer.pad_token_id,
    #     vocab_size=len(processor.tokenizer),
    # )
    
    # Create classifier and full SER model
    classifier = MLP_EmotionClassifier(input_dim, hidden_dim, num_classes)
    ser_model = SERModel(xlsr_model, classifier, layer_to_extract=12)  # Layer 12 is often optimal

    def predict_emotion(filepath):
        # Load and process the audio
        waveform = utils.PreProcess.load_audio(filepath)
        input_values = ser_model(waveform, return_tensors="pt", sampling_rate=16000).input_values

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

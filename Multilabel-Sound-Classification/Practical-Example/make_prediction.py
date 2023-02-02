import torch
import torchaudio
import numpy as np

from custom_dataset_dataloader import ConstructionVehicleSoundDataset
from cnn_sound_model import CNNSoundRecognizer
from train_test_loop import ANNOTATIONS_FILE, AUDIO_DIR, target_num_samples, target_sample_rate


# Create Function for prediction
def predict(model, input, target_tensor, target_string, threshold=None):
    model.eval()
    with torch.inference_mode():
        predictions = model(input).squeeze()
        # Tensor (1, 14) -> [ [0.1, 0.01, ..., 0.6] ]
        predictions = torch.sigmoid(predictions)
        predictions_percentage = predictions * 100
        # Actual Label as Tensor
        target_tensor = target_tensor
        # Actual Label as String
        target_string = target_string

        for label, prediction in zip(target_string, predictions_percentage):
            if prediction > threshold:
                print(f"{label}: {prediction:.2f}%")

    return predictions, target_tensor, target_string


# Run and Predict
if __name__ == "__main__":
    # Specified device agnostic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    # load our saved model
    cnn = CNNSoundRecognizer().to(device)
    state_dict = torch.load("cnn_sound_activities_recognizer.pth")
    cnn.load_state_dict(state_dict)

    # Audio Transformation (Mel Spectrogram)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # Create Custom Audio Dataset Object
    construction_audio_dataset = ConstructionVehicleSoundDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=mel_spectrogram,
        target_sample_rate=target_sample_rate,
        target_num_samples=target_num_samples,
        device=device
    )

    # get a sample from the urban sound dataset for inference
    input, target_tensor, target_string = construction_audio_dataset[45][0], \
        construction_audio_dataset[45][1], \
        construction_audio_dataset[45][2]  # [batch size, num_channels, fr, time]
    input = input.unsqueeze_(0)

    # make an inference
    predicted_tensor, actual_tensor, actual_label = predict(cnn, input, target_tensor, target_string, 0.6)
    print(f"\n Predicted Tensor: '{predicted_tensor}',"
          f"\n Actual Tensor: '{actual_tensor}',"
          f"\n Actual Label: '{actual_label}'")

# Custom Dtaset and Data Loader

# Import Necessary Libraries
import os
import torch
import pandas as pd
import torchaudio
import numpy as np
from torch.utils.data import Dataset

# Create a class for creating multilabel dataset
class ConstructionVehicleSoundDataset():
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 target_num_samples,
                 device):

        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.target_num_samples = target_num_samples
        self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label_as_tensor, label_as_string = self._get_audio_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_needed(signal, sr)
        signal = self._same_audio_channel_if_needed(signal)
        signal = self._cut_num_samples_if_needed(signal)
        signal = self._padding_if_needed(signal)
        signal = self.transformation(signal)

        return signal, label_as_tensor, label_as_string

    # Function for getting audio smaple path
    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path

    # Function for getting corresponding label of each audio file
    def _get_audio_label(self, index):
        label_as_tensor = self.annotations.iloc[index, 1:]
        label_as_tensor = torch.tensor(label_as_tensor.values.astype(np.float32))
        label_as_string = self.annotations.columns.tolist()[1:]
        return label_as_tensor, label_as_string

    # Function for resampling of the original sample rate of each audio file and make uniform
    def _resample_if_needed(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    # Function for converting multiple channel to single channel
    def _same_audio_channel_if_needed(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    # This function for uniform number of sample for all audio signal when signal ample >expected samples
    def _cut_num_samples_if_needed(self, signal):
        if signal.shape[1] > self.target_num_samples:
            signal = signal[:, :self.target_num_samples]
        return signal

    # # This function for uniform number of sample for all audio signal when signal ample < expected samples
    def _padding_if_needed(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.target_num_samples:
            num_missing_samples = self.target_num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


if __name__ == "__main__":
    ANNOTATIONS_FILE = r"C:\Users\user\Documents\DeepLearning\ProjectAudio\PytorchforAudio\Final Arrangement\Small data\metadata\vehicle_multilabel_metadata.csv"
    AUDIO_DIR = r"C:\Users\user\Documents\DeepLearning\ProjectAudio\PytorchforAudio\Final Arrangement\Small data\audio"

    target_sample_rate = 22050  # For specified same sample rate all audio
    target_num_samples = 22050  # We want at leat 1 sec duration # Duration = Num Samples / Sample Rate

    # Specified device agnostic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    # Audio Transformation (Mel Spectrogram)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # Create Custom Audio Dataset Object
    construction_equipment_sound = ConstructionVehicleSoundDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=mel_spectrogram,
        target_sample_rate=target_sample_rate,
        target_num_samples=target_num_samples,
        device=device
    )

    print(f"There are {len(construction_equipment_sound)} samples in the dataset.")

    # Let's check our first audio information
    signal, label_as_tensor, label_as_string = construction_equipment_sound[0]

    print(signal)
    print(label_as_tensor)
    print(label_as_string)

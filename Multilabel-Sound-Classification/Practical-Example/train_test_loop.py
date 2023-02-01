# Import Libraries
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from cnn_sound_model import CNNSoundRecognizer
from custom_dataset_dataloader import ConstructionVehicleSoundDataset
from torch.utils.data import random_split
from timeit import default_timer as timer

# Set required parameter and directory for training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

ANNOTATIONS_FILE = r"C:\Users\user\Documents\DeepLearning\ProjectAudio\PytorchforAudio\Vehicle Sound Classification\Small data\metadata\vehicle_multilabel_metadata.csv"
AUDIO_DIR = r"C:\Users\user\Documents\DeepLearning\ProjectAudio\PytorchforAudio\Vehicle Sound Classification\Small data\audio"
target_sample_rate = 22050  # For specified same sample rate all audio
target_num_samples = 22050  # We want at leat 1 sec duration # Duration = Num Samples / Sample Rate


# Create a function to time our experiment
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """Prints difference between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# Create a Dataloader function
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


# ## Train Loop Function
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):
    # Set initial loss
    train_loss = 0
    # Put model into training mode
    model.train()
    for batch, (X, y, z) in enumerate(data_loader):
        # Send to the GPU
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_pred = model(X)
        # 2. Loss calculation
        loss = loss_fn(y_pred, y)
        train_loss += loss
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss Backward
        loss.backward()
        # 5. Optimizer Step
        optimizer.step()
    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    print(f"Train Loss: {train_loss: .4f}")


# Test Loop Function
def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device):
    # Initial loss and accuaracy
    test_loss = 0
    # Put the model into evaluation mode
    model.eval()
    # Open and make prediction with inference mode
    with torch.inference_mode():
        for X, y, z in data_loader:
            # Send data to the device
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred = model(X)
            # Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)

        # Adjust average loss and accuracy
        test_loss /= len(data_loader)
        print(f"Test Loss: {test_loss: .4f}")


# Training for all epoch
def train(model, train_data_loader, test_data_loader, loss_fn, optimizer, device, epochs):
    # Set manual seed
    torch.manual_seed((42))
    # Set time function to get start time
    train_time_start_on_device = timer()
    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n----------")

        # Train step
        train_step(model=model,
                   data_loader=train_data_loader,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   device=device)

        # Test Step
        test_step(model=model,
                  data_loader=test_data_loader,
                  loss_fn=loss_fn,
                  device=device)
        print("---------------------------")

    # Set function to grt end time
    train_time_end_on_device = timer()
    # Print total time
    total_train_time = print_train_time(start=train_time_start_on_device,
                                        end=train_time_end_on_device,
                                        device=device)

    print(f"Total Run Time: {total_train_time}")
    print("Finished training")


if __name__ == "__main__":
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
    datasetv1 = ConstructionVehicleSoundDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=mel_spectrogram,
        target_sample_rate=target_sample_rate,
        target_num_samples=target_num_samples,
        device=device
    )
    # Split train and val dataset
    num_samples = len(datasetv1)
    num_train_samples = round(num_samples * 0.8)
    num_test_samples = num_samples - num_train_samples
    train_dataset, test_dataset = random_split(datasetv1, [num_train_samples, num_test_samples])
    # print(train_dataset[0], test_dataset[0])

    # Create training and validation data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Let's check out what what we've created
    # print(f"DataLoaders: {train_dataloader, test_dataloader}")
    # print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}...")
    # print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}...")

    # Check out what's inside the training dataloader
    train_features_batch, train_labels_tensor_batch, train_labels_tensor_string = next(iter(train_dataloader))
    # test_features_batch, test_labels_batch = next(iter(test_dataloader))
    print(train_features_batch.shape, train_labels_tensor_batch.shape)
    # print(test_features_batch.shape, test_labels_batch.shape)
    #
    # # Create a flatten layer
    # flatten_model = nn.Flatten()
    #
    # # Get a single sample
    # x = train_features_batch[0]
    # print(x, x.shape)
    #
    # # Let's Flatten the sample
    # output = flatten_model(x)  # perform forward pass
    # output, output.shape

    cnn_MODELv1 = CNNSoundRecognizer().to(device)
    print(cnn_MODELv1)
    #
    # # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_MODELv1.parameters(),
                                 lr=LEARNING_RATE)
    # train model
    train(cnn_MODELv1, train_dataloader, test_dataloader, loss_fn, optimizer, device, EPOCHS)

    # save model
    torch.save(cnn_MODELv1.state_dict(), "cnn_sound_activities_recognizer.pth")
    print("Trained feed forward net saved at cnn_sound_activities_recognizer.pth")

"""
In this file, we create and train convolutional autoencoder for audio data.
"""

import torch
import torchaudio
import torchinfo

class AudioConvAE(torch.nn.Module):
    """
    This PyTorch module contains the definition
    of the convolutional autoencoder.
    It consists of an encoder and a decoder, each made of convolutional layers.
    Since this is an autoencoder, we want to reduce the dimensionality of the signal
    in the latent space. We do this through feature extraction with 2-D convolutional
    layers followed by max pooling, which reduces the dimensionality by 2.

    In the decoder, we expand the dimensionality back up from the latent space,
    using transposed convolutions to "undo" the convolutions in the encoder.

    In "classical" audio feature extraction, it's common to use the "delta"
    and "delta-delta" features from the mel-frequency cepstral coefficients.
    Computing those require three consecutive values in the signal,
    to compute the central finite differences.
    Here, we use a kernel of size 3, which allows the model to learn
    features similar to the delta and delta-delta values,
    but that perhaps might be even better for representing the salient
    features of the signal.
    """
    def __init__(self):
        super(AudioConvAE, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train(model: torch.nn.Module, input_data: torch.Tensor) -> None:
    """
    This function trains the model.

    We use 100 iterations of a single data sample for demonstration purposes.

    We use the ADAMW optimizer and since we want to train an autoencoder,
    we use the mean squared error loss function
    so that the model learns to return the input it was given as output.
    """
    n_iters = 100

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for i in range(n_iters):
        # zero the gradient in the optimizer
        optimizer.zero_grad()
        # run the model forwards
        output_data = model(input_data)
        # compute the mean-squared error loss
        loss = criterion(output_data, input_data)
        # compute the gradients
        loss.backward()
        # update the parameters of the model
        optimizer.step()
        print(f"Iteration: {i}, Loss: {loss}")


def load_and_preprocess() -> torch.Tensor:
    """
    Load and preprocess the input data.
    1. Load the .wav file.
    2. Resample to 16000 Hz.
    3. Compute the Mel spectrogram.
    4. Log scale and rescale.
    """
    audio, sample_rate = torchaudio.load("./epi.wav")  # tensor, int

    # Resample the audio to 16000 Hz,
    # then compute the Mel-frequency spectrogram.
    transforms = torch.nn.Sequential(
        torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16e3, dtype=audio.dtype
        ),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16e3,
            win_length=int(23 * 16),
            hop_length=int(10 * 16),
            n_mels=80,
        ),
    )

    mel_spec: torch.Tensor = transforms(audio)  # (channel, n_mels, time)

    # This rescaling preserves the "distance" between values,
    # but squashes the spectrogram to be between -1.4 and 1.4 or so.
    # This makes sure that the outputs from any given layer of the model
    # aren't too big or too small.
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    # Cut off the last row so that the last dimension has an even
    # number of elements.
    # Otherwise, MaxPooling will break.
    log_spec = log_spec[:, :, 0:-1]

    return log_spec


if __name__ == "__main__":
    input_data = load_and_preprocess()
    model = AudioConvAE()
    torchinfo.summary(model, input_data=input_data)
    train(model, input_data)

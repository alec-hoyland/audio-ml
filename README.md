# audio-ml
Demos of supervised and unsupervised machine learning for audio in Python

## audio_analysis.ipynb

This notebook describes how to visualize and analyze audio signals using supervised and unsupervised machine learning methods.

We explore two audio files, one containing speech and music and the other just speech.
For both, we visualize the waveforms, spectrograms, and compute the log-Mel coefficients.

We build an unsupervised machine learning model for detecting regions of speech using denoising algorithms,
Mel-frequency cepstral coefficients, dimensionality reduction using UMAP, and clustering using K-Means.

We also run the pretrained Whisper model from WhisperCPP on the second signal to transcribe the audio.

## autoencoder.py

This script shows how to build and train a convolutional autoencoder for audio data.

The model consists of an encoder and a decoder, each made of convolutional layers.
Since this is an autoencoder, we want to reduce the dimensionality of the signal
in the latent space. We do this through feature extraction with 2-D convolutional
layers followed by max pooling, which reduces the dimensionality by 2.

In the decoder, we expand the dimensionality back up from the latent space,
using transposed convolutions to "undo" the convolutions in the encoder.
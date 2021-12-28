# Autoencoder in Pytorch

In this repository there are a series of Computer Vision related projects which use AutoEncoder type arhitectures.

## Image Denosing 

Demo project that denoises an image in 2 steps. 

1. Pretraining - training the AE to reconstruct MNIST digits.
2. With the decoder fixed, the encoder is trained to match the latent space of a clean image

[Project source code](https://github.com/DariusCatrina/Autoencoder-Projects-in-Pytorch/tree/main/ImageDenoising)

## Image Colorization

Demo project that attempts to colorize 128by128 images.

The Colorization Netowrk is an Autoencoder with:

1. VGG19/16 pretrained network as an encoder to extract important features(using transfer learning)
2. Custom decoder with 7 CNN Block (Conv Layer, BatchNormalization + ReLU/TanH activation)



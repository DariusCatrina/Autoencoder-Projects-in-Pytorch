from DataSets import  test_loader
from Model import CONV_AutoEncoder, device
import numpy as np
import matplotlib.pyplot as plt
import torch


reconstrct_autoencoder = CONV_AutoEncoder()
reconstrct_autoencoder.load_state_dict(torch.load('model_reconstruction.pth'))
reconstrct_autoencoder.to(device)
reconstrct_autoencoder.eval()

denoised_autoencoder = CONV_AutoEncoder()
denoised_autoencoder.load_state_dict(torch.load('model_denoising.pth'))
denoised_autoencoder.to(device)
denoised_autoencoder.eval()

noise_factor = 0.95
def test_denoise(idx=20):
    for i, (target, _) in enumerate(test_loader):
        if i==idx:
            denoised_autoencoder.zero_grad()

            data = target + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=target.shape)
            data = np.clip(data, 0, 1)
            data, target = data.float().to(device), target.float().to(device)

            output = denoised_autoencoder.forward(data)
            data, target, output = data.cpu().detach().numpy(), target.cpu().detach().numpy(), output.cpu().detach().numpy()
            

            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(data[14,0,:,:], cmap='gray')
            axs[1].imshow(target[14,0,:,:], cmap='gray')
            axs[2].imshow(output[14,0,:,:], cmap='gray')

            axs[0].title.set_text('Noised Image')
            axs[1].title.set_text('Clean Image')
            axs[2].title.set_text('Denoised Image(Ours)')

            plt.show()

def test_reconstruct(idx=25):
    for i, (data, _) in enumerate(test_loader):
        if i==idx:
            reconstrct_autoencoder.zero_grad()

            output = reconstrct_autoencoder.forward(data)
            data, output = data.cpu().detach().numpy(), output.cpu().detach().numpy()
            
            _, axs = plt.subplots(1, 2)
            axs[0].imshow(data[14,0,:,:], cmap='gray')
            axs[1].imshow(output[14,0,:,:], cmap='gray')

            axs[0].title.set_text('Input Image')
            axs[1].title.set_text('Reconstructed Image(Ours)')

            plt.show()

test_reconstruct(29)
test_denoise(29)
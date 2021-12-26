from DataSets import train_loader, test_loader
from Model import autoencoder, optimizer, MSE, device
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

# Firstly, we train the model to reconstruct the clean images
def train_for_image_reconstruction(epochs):
    autoencoder.train()
    train_losses = []

    _range = tqdm(range(epochs))
    for epoch in _range:
        train_loss = 0
        for _, (data, _) in enumerate(train_loader):
            autoencoder.zero_grad()

            data = data.float().to(device)
            output = autoencoder.forward(data)

            loss = MSE(output, data)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss/len(train_loader))

        if epoch % 5 == 0:
            torch.save(autoencoder.state_dict(), './model_reconstruction.pth')
            
        if epoch >= 6:
            mean_loss = sum(train_losses[epoch-6:epoch])/6
            if mean_loss > train_losses[epoch-6]:
                print("Training loss has not decreased in the last 6 epochs. Stopping training.")
                return train_losses
        
        _range.set_description(f"Loss for the epoch {epoch} is {train_losses[-1]}")

    return train_losses

def plot_img_reconstruction_loss(train_losses):
    import matplotlib.pyplot as plt
    plt.plot(train_losses, np.linspace(len(train_losses)))
    plt.xlabel("Image Reconstruction - Training Epochs")
    plt.ylabel("Losses over each epoch") 
    plt.show()


# Secoundly, we train the model to clean the noise images via freezing the parameters 
# of the decoder and retraining the encoder
noise_factor = 0.95

def train_for_denoising(epochs):
    autoencoder.train()
    autoencoder.decoder.requires_grad = False
    train_losses = []

    _range = tqdm(range(epochs))
    for epoch in _range:
        train_loss = 0
        for _, (target, _) in enumerate(train_loader):
            autoencoder.zero_grad()
            #adding noise to the images
            data = target + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=target.shape)
            data = np.clip(data, 0, 1)
            data, target = data.float().to(device), target.float().to(device)

            output = autoencoder.forward(data)

            loss = MSE(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss/len(train_loader))

        if epoch % 5 == 0:
            torch.save(autoencoder.state_dict(), './model_denoising.pth')
            
        if epoch >= 6:
            mean_loss = sum(train_losses[epoch-6:epoch])/6
            if mean_loss > train_losses[epoch-6]:
                print("Training loss has not decreased in the last 6 epochs. Stopping training.")
                return train_losses

        _range.set_description(f"Loss for the epoch {epoch} is {train_losses[-1]}")
    
    return train_losses

def plot_img_denoising_loss(train_losses):
    import matplotlib.pyplot as plt
    plt.plot(train_losses, np.range(len(train_losses)))
    plt.xlabel("Image Denoising - Training Epochs")
    plt.ylabel("Losses over each epoch") 
    plt.show()

if __name__ == "__main__":
    model_saved = True
    if model_saved:
        autoencoder.load_state_dict(torch.load('./model_denoising.pth'))
    else:
        print('Firstly, we train the model to reconstruct the clean images')
        train_losses = train_for_image_reconstruction(100)
        print('Secoundly, we train the model to clean the noise images')
        denoising_losses = train_for_denoising(100)

        plot_img_reconstruction_loss(train_losses)
        plot_img_denoising_loss(denoising_losses)


    # Test
    autoencoder.eval()
    for i, (target, _) in enumerate(test_loader):
      if i==58:
        autoencoder.zero_grad()

        data = target + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=target.shape)
        data = np.clip(data, 0, 1)
        data, target = data.float().to(device), target.float().to(device)

        output = autoencoder.forward(data)
        data, target, output = data.cpu().detach().numpy(), target.cpu().detach().numpy(), output.cpu().detach().numpy()
        

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(data[14,0,:,:], cmap='gray')
        axs[1].imshow(target[14,0,:,:], cmap='gray')
        axs[2].imshow(output[14,0,:,:], cmap='gray')

        axs[0].title.set_text('Noised Image')
        axs[1].title.set_text('Clean Image')
        axs[2].title.set_text('Denoised Image(Ours)')

        plt.show()






      





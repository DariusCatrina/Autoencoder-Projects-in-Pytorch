import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cuda:0":
    print("Using the GPU")
else:
    print("Using the CPU")

class CONV_AutoEncoder(nn.Module):
  def __init__(self):
    super(CONV_AutoEncoder, self).__init__()
    self.encoder = nn.Sequential(         
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),                              
            nn.ReLU(),                      
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),                      
            nn.MaxPool2d((2,2)),
        ) #first it will be trained the encode clean image and, in the secound training to encode the noise image

    self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),                      
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2),        
        )  #it will be trained to decode the images

    self.sigmoid = nn.Sigmoid()
    
  def forward(self, data):
    x = self.encoder(data)
    x = self.decoder(x)
    
    return self.sigmoid(x)


autoencoder = CONV_AutoEncoder()
autoencoder.to(device)

MSE = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

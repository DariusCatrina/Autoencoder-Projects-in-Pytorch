#AutoEncoder Arhitecture : Encoder + Decoder 
#Encoder: VGG 16 pre-trained / ResNet
#Decoder: Custum decoder with 3x CNN Blocks

########################################### VGG Model ###########################################


import torch.nn as nn
import torch.hub



VGG_16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
VGG_19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

VGG_19bn_url = 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
VGG_16bn_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
VGG_19_url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
VGG_16_url = "https://download.pytorch.org/models/vgg16-397923af.pth"

def creat_CNNBlock(in_channels, out_channels, kernel_size, use_bn, use_act=True,**kwargs):
  CNNBlock = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, **kwargs),  
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),  
            nn.ReLU(inplace=True) if use_act else nn.Identity(),               
        ]
  
  return CNNBlock

class VGG(nn.Module):
  def __init__(self, config, url, use_bn=True):
    super().__init__()
    self.config = config
    self.use_bn = use_bn
    self.model = self.create_model()
    if url:
      self.load_pretrained_weights(url)
    
  
  def create_model(self):
    in_channels = 1
    self.layers = []
    for l in self.config:
      if l == 'M':
          self.layers.extend([nn.MaxPool2d(kernel_size=2, stride=2)])
      else:
          self.layers.extend(creat_CNNBlock(in_channels, l, kernel_size=3, use_bn=self.use_bn, padding=1))
          in_channels = l

    return nn.Sequential(*self.layers)

  def load_pretrained_weights(self, url):
    state_dict = torch.hub.load_state_dict_from_url(url)
    self.model.load_state_dict(state_dict, strict=False)

  def forward(self, input):
    return self.model(input)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vgg = VGG(VGG_19_config, VGG_19_url, use_bn=False)
vgg.to(device)

# VGG output: torch.Size([16, 512, 4, 4])
## DECODER: conv block: [conv(512 -> 128), batch norm, relu] ± UpSample

########################################### Decoder Model ###########################################

def creat_CNNBlock(in_channels, out_channels, kernel_size, use_bn, use_act='relu',**kwargs):
  #print(in_channels, out_channels, kernel_size)
  CNNBlock = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, **kwargs),  
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),               
        ]
  if use_act=='relu':
      CNNBlock += [nn.ReLU(inplace=True)]
  elif use_act=='tanh':
      CNNBlock += [nn.Tanh()]
  else:
    return CNNBlock
  return CNNBlock

  

decoder_config = [256, 128, 'U', 64, 'U', 32, 'U', 16, 'U', 3, 'U']
act = 5*['relu'] + ['tanh']
bn = 5*[True] + [False]

class Decoder(nn.Module):
  def __init__(self, config, use_bn) -> None:
      super().__init__()
      self.config = config
      self.bn = use_bn
      self.model = self.create_model()


  def create_model(self):
    in_channels = 512
    self.layers = []
    idx = 0
    for l in self.config:
      if l == 'U':
          self.layers.extend([nn.Upsample(scale_factor=2)])
      else:
          self.layers.extend(creat_CNNBlock(in_channels, l, kernel_size=3, use_act=act[idx], use_bn=self.bn[idx], padding=1))
          in_channels = l
          idx+=1

    return nn.Sequential(*self.layers)

  def forward(self, input):
    return self.model(input)
    
decoder = Decoder(decoder_config, use_bn=bn)
decoder.to(device)

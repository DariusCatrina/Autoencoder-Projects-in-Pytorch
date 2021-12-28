import torch
import torchvision.transforms as T
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm
from models import vgg, decoder, device
from dataset import train_dataloader, test_dataloader

for param in vgg.parameters():
  param._requier_grad = False


to_pil_transform = T.Compose([
    T.ToPILImage()
])


optimizer = Adam(decoder.parameters(), lr=0.001)
mse_loss = MSELoss()


def train_one_epoch(epoch): # returns the mean loss over the specific epoch  
  total_loss = 0
  print(f'Training... epoch {epoch}')
  loop = tqdm(train_dataloader, leave=True)
  decoder.model.train()

  for idx, batch in enumerate(loop):

    batch = [sample.to(device) for sample in batch]

    gray_input, rgb_target, ab_target = batch

    decoder.model.zero_grad()
    ab_output = decoder.forward(vgg.forward(gray_input))

    loss = mse_loss(ab_output, ab_target)
    total_loss+=loss

    #backpropagation 
    loss.backward()
    optimizer.step()

    #gpu memory cleaning 
    loss = loss.detach().cpu().numpy()
    ab_output = ab_output.detach().cpu().numpy()
    batch = [sample.detach().cpu().numpy() for sample in batch]

    loop.set_description(f'Loss for the epoch {epoch}: {loss}')



  return total_loss/len(train_dataloader)

def test_one_epoch(epoch): # returns the mean loss over the specific epoch  
  total_loss = 0
  print(f'Testing... epoch {epoch}')
  decoder.model.eval()
  loop = tqdm(test_dataloader, leave=True)

  for idx, batch in enumerate(loop):

    batch = [sample.to(device) for sample in batch]

    gray_input, rgb_target, ab_target = batch

    with torch.no_grad():
      decoder.model.zero_grad()
      ab_output = decoder.forward(vgg.forward(gray_input))

      loss = mse_loss(ab_output, ab_target)
      loss = loss.detach().cpu().numpy()
      total_loss+=loss

      ab_output = ab_output.detach().cpu().numpy()  
      batch = [sample.detach().cpu().numpy() for sample in batch]

    loop.set_description(f'Loss for the epoch {epoch}: {loss}')


  return total_loss/len(test_dataloader)  



train_losses = [2.7]
test_losses = [2.6]

def train_for_image_colorization_AE(num_epochs):
  best_loss = float('inf')
  for epoch in range(1, num_epochs):

    print(f'EPOCH no {epoch}')

    train_loss = train_one_epoch(epoch)
    test_loss = test_one_epoch(epoch)
    print(train_loss, test_loss)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if test_loss < best_loss:
      torch.save(decoder.state_dict(), 'image_colorization_AE.pt') 
      best_loss = test_loss

    #overfitting

    if epoch >= 3:
      mean_loss = sum(train_losses[epoch-3:epoch])/3
      if mean_loss > train_losses[epoch-3]:
        print("Overfitting detected")


train_for_image_colorization_AE(10)

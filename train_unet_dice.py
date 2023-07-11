from .utils import Dataset, DiceLoss

import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

#wget

num_workers = 4
batch_size = 8
# Cria os data loader de treino e de teste
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)

test_loader = DataLoader(test, batch_size=batch_size, shuffle=True,
                         num_workers=num_workers, pin_memory=True)


test = Dataset('/content/dataset/test', (256,256))
train = Dataset('/content/dataset/train', (256,256))

# Baixa o modelo pré-treinado para identificação de tumor cerebral
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
model.cuda()

# Define as camadas a serem ajustadas
for param in model.parameters():
  param.requires_grad_ = False

model.decoder1.requires_grad_ = True
model.decoder2.requires_grad_ = True
model.decoder3.requires_grad_ = True
model.decoder4.requires_grad_ = True


#criterion1 = nn.BCEWithLogitsLoss()
#criterion2 = nn.MSELoss()
criterion_dice = DiceLoss()

optimizer = optim.Adam(model.parameters(), lr=2e-3)


epochs = 100
for epoch in range(epochs):
  model.train()
  print(epoch)
  running_loss = 0
  for iteration, data in enumerate(train_loader, start=1):
    img = Variable(data['img'].cuda())
    mask = Variable(data['mask'].cuda())
    #label = Variable(data['label'].cuda())

    optimizer.zero_grad()

    outputs = model(img)
    loss = criterion_dice(outputs, mask)

    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  epoch_loss = running_loss/len(train_loader)
  print(f'Epoch: {epoch + 1}/{epochs} - Loss: {epoch_loss: .4f}')

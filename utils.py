from PIL import Image, ImageOps
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms


torch.manual_seed(42)

# Classe para leitura das imagens
class Dataset(data.Dataset):
  def __init__(self, path, image_size, channel='RGB'):
    self.files_path = path
    self.imagens = []
    self.path_imagens = []
    self.masks = []
    self.path_masks = []
    self.label = []
    self.__transform(image_size)
    #Lista as classes dos dados
    classes = os.listdir(self.files_path)
    # Itera sobre as classes
    for c in classes:
      # Monta o path para a pasta da classe
      path_class = os.path.join(self.files_path, c)
      # Pega o nome dos arquivos
      nome_imagens = os.listdir(path_class)
      for nome_imagem in nome_imagens:
        if 'mask' in nome_imagem:
          continue
        # Monta o caminho para a imagem de entrada e a mácara
        path_imagem = os.path.join(path_class, nome_imagem)
        path_mask = path_imagem.replace('.', '_mask.')
        # Faz a leitura da máscara
        imagem = Image.open(path_imagem).convert(mode='RGB')
        mask = Image.open(path_mask)
        # Converte a máscara para escala de cinza
        mask = ImageOps.grayscale(mask)
        # Salva as imagens na lista
        self.imagens.append(imagem)
        self.path_imagens.append(path_imagem)
        self.masks.append(mask)
        self.path_masks.append(path_mask)
        self.label.append(c)

  def __transform(self, image_size):
    # Inicializa o transormador dos dados
    self.transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

  def __len__(self):
    return len(self.imagens)

  def __getitem__(self, idx):
    # Pega as imagens no índice
    imagem = self.imagens[idx]
    mask = self.masks[idx]
    classe = self.label[idx]
    # Transforma as imagens
    imagem = self.transform(imagem)
    mask = self.transform(mask)
    # Retorna os dados das imagens
    return {'img': imagem, 'mask': mask, 'label': classe}

# Fonte: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
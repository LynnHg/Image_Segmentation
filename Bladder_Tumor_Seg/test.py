import torch

from models_pkg.Unet import Unet

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# model = Unet(1, 2)
# model = model.to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# x = torch.randn([3,1,160, 160])
# x = x.to(device)
model = Unet(1, 2)
model = model.to(device)

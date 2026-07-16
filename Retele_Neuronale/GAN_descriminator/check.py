import torch

torch.backends.cudnn.enabled = False

from model import GANDiscriminator


DEVICE="cuda" if torch.cuda.is_available() else "cpu"


model=GANDiscriminator().to(DEVICE)


x=torch.randn(

    32,

    3,

    224,

    224

).to(DEVICE)


y=model(x)


print()

print("Input :",x.shape)

print("Output:",y.shape)

print()

print("Clase:",2)

print()

print("Model OK!")

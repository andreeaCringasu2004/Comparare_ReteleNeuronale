import torch

from generator import Generator
from discriminator import Discriminator


G=Generator()

D=Discriminator()


noise=torch.randn(32,100)


fake=G(noise)


print(
    "Generator:",
    fake.shape
)


out=D(fake)


print(
    "Discriminator:",
    out.shape
)

import os
import torch

torch.backends.cudnn.enabled = False

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from generator import Generator
from discriminator import Discriminator



# =====================
# CONFIG
# =====================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


EPOCHS = 50

BATCH_SIZE = 32

LR = 0.0002

NOISE_DIM = 100

PATIENCE = 5



torch.backends.cudnn.enabled = False



print(
    "Device folosit:",
    DEVICE
)



# =====================
# PATH DATASET
# =====================


BASE_PATH = os.path.expanduser(
    "~/deepfake_env/Comparare_ReteleNeuronale/videos/Celeb-DF"
)



TRAIN_PATH = os.path.join(
    BASE_PATH,
    "train"
)



VAL_PATH = os.path.join(
    BASE_PATH,
    "validation"
)



# =====================
# TRANSFORM
# =====================


transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),


    transforms.ToTensor(),


    transforms.Normalize(

        [0.5,0.5,0.5],

        [0.5,0.5,0.5]

    )

])



# =====================
# DATASET
# =====================


train_dataset = datasets.ImageFolder(

    TRAIN_PATH,

    transform

)



val_dataset = datasets.ImageFolder(

    VAL_PATH,

    transform

)



train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=4

)



val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=4

)



print("\nClase:")

print(train_dataset.classes)


print(
    "Imagini train:",
    len(train_dataset)
)


print(
    "Imagini validation:",
    len(val_dataset)
)



# =====================
# MODELE
# =====================


generator = Generator(

    NOISE_DIM

).to(DEVICE)



discriminator = Discriminator().to(DEVICE)



criterion = torch.nn.BCELoss()



optimizer_G = torch.optim.Adam(

    generator.parameters(),

    lr=LR,

    betas=(0.5,0.999)

)



optimizer_D = torch.optim.Adam(

    discriminator.parameters(),

    lr=LR,

    betas=(0.5,0.999)

)



scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer_D,

    mode="max",

    factor=0.5,

    patience=2

)



best_acc = 0

counter = 0



# =====================
# TRAIN
# =====================


for epoch in range(EPOCHS):


    generator.train()

    discriminator.train()



    loss_D_total = 0

    loss_G_total = 0


    d_real_total = 0

    d_fake_total = 0



    for images,_ in train_loader:


        images = images.to(DEVICE)



        batch_size = images.size(0)



        real_labels = torch.ones(

            batch_size,

            1,

            device=DEVICE

        )



        fake_labels = torch.zeros(

            batch_size,

            1,

            device=DEVICE

        )



        # =====================
        # TRAIN DISCRIMINATOR
        # =====================


        optimizer_D.zero_grad()



        real_output = discriminator(

            images

        )



        loss_real = criterion(

            real_output,

            real_labels

        )



        noise = torch.randn(

            batch_size,

            NOISE_DIM,

            device=DEVICE

        )



        fake_images = generator(

            noise

        )



        fake_output = discriminator(

            fake_images.detach()

        )



        loss_fake = criterion(

            fake_output,

            fake_labels

        )



        loss_D = (

            loss_real +

            loss_fake

        )



        loss_D.backward()

        optimizer_D.step()



        # =====================
        # TRAIN GENERATOR
        # =====================


        optimizer_G.zero_grad()



        fake_output = discriminator(

            fake_images

        )



        loss_G = criterion(

            fake_output,

            real_labels

        )



        loss_G.backward()

        optimizer_G.step()



        loss_D_total += loss_D.item()

        loss_G_total += loss_G.item()



        d_real_total += real_output.mean().item()

        d_fake_total += fake_output.mean().item()



    # =====================
    # VALIDATION DISCRIMINATOR
    # =====================


    discriminator.eval()


    correct = 0

    total = 0



    with torch.no_grad():


        for images, labels in val_loader:


            images = images.to(DEVICE)


            labels = labels.to(DEVICE)



            outputs = discriminator(

                images

            )



            predictions = (

                outputs > 0.5

            ).long().squeeze()



            correct += (

                predictions == labels

            ).sum().item()



            total += labels.size(0)



    val_acc = correct / total



    scheduler_D.step(

        val_acc

    )



    current_lr = optimizer_D.param_groups[0]["lr"]



    print(
        f"""
=========================
Epoch {epoch+1}/{EPOCHS}

Discriminator Loss:
{loss_D_total/len(train_loader):.4f}

Generator Loss:
{loss_G_total/len(train_loader):.4f}


D(real):
{d_real_total/len(train_loader):.4f}

D(fake):
{d_fake_total/len(train_loader):.4f}


Validation Accuracy:
{val_acc:.4f}


Learning Rate:
{current_lr}

=========================
"""
    )



    # =====================
    # SAVE BEST
    # =====================


    if val_acc > best_acc:


        best_acc = val_acc

        counter = 0



        torch.save(

            discriminator.state_dict(),

            "best_discriminator.pth"

        )



        torch.save(

            generator.state_dict(),

            "best_generator.pth"

        )



        print(
            "Model nou salvat!"
        )


    else:


        counter += 1


        print(

            f"Fara imbunatatire: {counter}/{PATIENCE}"

        )



    if counter >= PATIENCE:


        print(
            "Early stopping activat!"
        )

        break



print("\nAntrenare GAN terminata!")


print(

    "Cea mai buna accuracy:",

    best_acc

)


print(

    "Discriminator:",

    "best_discriminator.pth"

)


print(

    "Generator:",

    "best_generator.pth"

)

import os
import torch

torch.backends.cudnn.enabled = False

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import VideoFrameDataset
from model import GANDiscriminatorLSTM



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


EPOCHS = 50
BATCH_SIZE = 8
SEQ_LEN = 10
LR = 0.001
PATIENCE = 5



print("Device folosit:", DEVICE)



BASE_PATH = os.path.expanduser(
    "~/deepfake_env/Comparare_ReteleNeuronale/videos/Celeb-DF"
)



transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]

    )

])



train_dataset = VideoFrameDataset(

    os.path.join(
        BASE_PATH,
        "train"
    ),

    SEQ_LEN,

    transform

)



validation_dataset = VideoFrameDataset(

    os.path.join(
        BASE_PATH,
        "validation"
    ),

    SEQ_LEN,

    transform

)



train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=4,

    pin_memory=True

)



validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=4,

    pin_memory=True

)



print()

print("Clase:")

print(train_dataset.classes)



print(
    "Secvente train:",
    len(train_dataset)
)


print(
    "Secvente validation:",
    len(validation_dataset)
)



model = GANDiscriminatorLSTM().to(DEVICE)



criterion = torch.nn.CrossEntropyLoss()



optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LR

)



scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=3

)



best_acc = 0

counter = 0



for epoch in range(EPOCHS):


    model.train()


    running_loss = 0

    correct = 0

    total = 0



    for videos, labels in train_loader:


        videos = videos.to(DEVICE)

        labels = labels.to(DEVICE)



        optimizer.zero_grad()



        outputs = model(
            videos
        )



        loss = criterion(

            outputs,

            labels

        )



        loss.backward()



        optimizer.step()



        running_loss += loss.item()



        predictions = torch.argmax(

            outputs,

            dim=1

        )



        correct += (

            predictions == labels

        ).sum().item()



        total += labels.size(0)



    train_loss = running_loss / len(train_loader)


    train_acc = correct / total



    model.eval()


    val_correct = 0

    val_total = 0



    with torch.no_grad():


        for videos, labels in validation_loader:


            videos = videos.to(DEVICE)

            labels = labels.to(DEVICE)



            outputs = model(
                videos
            )



            predictions = torch.argmax(

                outputs,

                dim=1

            )



            val_correct += (

                predictions == labels

            ).sum().item()



            val_total += labels.size(0)



    val_acc = val_correct / val_total



    scheduler.step(
        val_acc
    )



    current_lr = optimizer.param_groups[0]["lr"]



    print()

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )


    print(
        f"Loss: {train_loss:.4f}"
    )


    print(
        f"Train Accuracy: {train_acc:.4f}"
    )


    print(
        f"Validation Accuracy: {val_acc:.4f}"
    )


    print(
        f"Learning Rate: {current_lr:.6f}"
    )



    if val_acc > best_acc:


        best_acc = val_acc

        counter = 0



        torch.save(

            model.state_dict(),

            "best_gan_lstm.pth"

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


        print()

        print(
            "Early stopping activat!"
        )

        break



print()

print(
    "Antrenare terminata!"
)


print()

print(
    "Cea mai buna accuracy:",
    f"{best_acc:.4f}"
)


print(
    "Model:",
    "best_gan_lstm.pth"
)

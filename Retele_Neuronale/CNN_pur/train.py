import os
import json
import torch

torch.backends.cudnn.enabled = False

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNNPur


# =========================
# CONFIGURARE
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 50

LEARNING_RATE = 0.001

PATIENCE = 5


print("Device folosit:", DEVICE)


# =========================
# CAI DATASET
# =========================

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



# =========================
# TRANSFORMARI IMAGINI
# =========================


transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])



# =========================
# DATASET
# =========================


train_dataset = datasets.ImageFolder(

    root=TRAIN_PATH,

    transform=transform
)



validation_dataset = datasets.ImageFolder(

    root=VAL_PATH,

    transform=transform
)



print("\nClase:")
print(train_dataset.classes)


print(
    "Imagini train:",
    len(train_dataset)
)


print(
    "Imagini validation:",
    len(validation_dataset)
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



# =========================
# MODEL
# =========================


model = CNNPur().to(DEVICE)



criterion = torch.nn.CrossEntropyLoss()



optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE
)



# scade LR daca validation accuracy nu mai creste

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2
)



# =========================
# EARLY STOPPING
# =========================


best_accuracy = 0

early_stop_counter = 0



history = {

    "train_loss": [],

    "train_accuracy": [],

    "validation_accuracy": [],

    "learning_rate": []

}



# =========================
# ANTRENARE
# =========================


for epoch in range(EPOCHS):


    print(
        f"\nEpoch {epoch+1}/{EPOCHS}"
    )


    # -----------------
    # TRAIN
    # -----------------

    model.train()


    running_loss = 0

    correct = 0

    total = 0



    for images, labels in train_loader:


        images = images.to(DEVICE)

        labels = labels.to(DEVICE)



        optimizer.zero_grad()



        outputs = model(images)



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

    train_accuracy = correct / total



    # -----------------
    # VALIDATION
    # -----------------

    model.eval()


    val_correct = 0

    val_total = 0



    with torch.no_grad():


        for images, labels in validation_loader:


            images = images.to(DEVICE)

            labels = labels.to(DEVICE)



            outputs = model(images)



            predictions = torch.argmax(
                outputs,
                dim=1
            )



            val_correct += (
                predictions == labels
            ).sum().item()



            val_total += labels.size(0)



    val_accuracy = val_correct / val_total



    # scheduler

    scheduler.step(
        val_accuracy
    )



    current_lr = optimizer.param_groups[0]["lr"]



    print(
        f"Loss: {train_loss:.4f}"
    )


    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )


    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )


    print(
        f"Learning Rate: {current_lr}"
    )



    # salvare istoric


    history["train_loss"].append(
        train_loss
    )


    history["train_accuracy"].append(
        train_accuracy
    )


    history["validation_accuracy"].append(
        val_accuracy
    )


    history["learning_rate"].append(
        current_lr
    )



    # -----------------
    # BEST MODEL
    # -----------------


    if val_accuracy > best_accuracy:


        best_accuracy = val_accuracy


        early_stop_counter = 0



        torch.save(

            model.state_dict(),

            "best_cnn_pur.pth"

        )



        print(
            "Model nou salvat!"
        )



    else:


        early_stop_counter += 1



        print(
            f"Fara imbunatatire: {early_stop_counter}/{PATIENCE}"
        )



    if early_stop_counter >= PATIENCE:


        print(
            "\nEarly stopping activat!"
        )


        break



# =========================
# SALVARE ISTORIC
# =========================


with open(
    "training_history.json",
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )



print("\nAntrenare terminata!")

print(
    "Cea mai buna accuracy:",
    best_accuracy
)

print(
    "Model:",
    "best_cnn_pur.pth"
)

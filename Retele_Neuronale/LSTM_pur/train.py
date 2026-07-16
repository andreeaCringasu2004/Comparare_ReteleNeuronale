import os
import json
import torch

torch.backends.cudnn.enabled = False

from torch.utils.data import DataLoader

from torchvision import datasets, transforms


from model import LSTMPur



DEVICE="cuda" if torch.cuda.is_available() else "cpu"


BATCH_SIZE=32

EPOCHS=50

LR=0.001

PATIENCE=5



torch.backends.cudnn.enabled=False



BASE_PATH=os.path.expanduser(
    "~/deepfake_env/Comparare_ReteleNeuronale/videos/Celeb-DF"
)



transform=transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]

    )

])



train_dataset=datasets.ImageFolder(

    os.path.join(BASE_PATH,"train"),

    transform

)



val_dataset=datasets.ImageFolder(

    os.path.join(BASE_PATH,"validation"),

    transform

)



train_loader=DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=4

)


val_loader=DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=4

)



print(train_dataset.classes)



model=LSTMPur().to(DEVICE)



criterion=torch.nn.CrossEntropyLoss()



optimizer=torch.optim.Adam(

    model.parameters(),

    lr=LR

)



scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2

)



best_acc=0

counter=0



history={

"train_loss":[],

"train_acc":[],

"val_acc":[],

"lr":[]

}



for epoch in range(EPOCHS):


    model.train()


    total=0

    correct=0

    loss_total=0



    for images,labels in train_loader:


        images=images.to(DEVICE)

        labels=labels.to(DEVICE)



        optimizer.zero_grad()


        outputs=model(images)


        loss=criterion(outputs,labels)


        loss.backward()


        optimizer.step()



        loss_total+=loss.item()



        pred=torch.argmax(

            outputs,

            dim=1

        )


        correct+=(pred==labels).sum().item()

        total+=labels.size(0)



    train_acc=correct/total

    train_loss=loss_total/len(train_loader)



    model.eval()


    correct=0

    total=0


    with torch.no_grad():


        for images,labels in val_loader:


            images=images.to(DEVICE)

            labels=labels.to(DEVICE)


            outputs=model(images)


            pred=torch.argmax(outputs,1)


            correct+=(pred==labels).sum().item()

            total+=labels.size(0)



    val_acc=correct/total



    scheduler.step(val_acc)



    lr=optimizer.param_groups[0]["lr"]



    print(
        f"""
Epoch {epoch+1}/{EPOCHS}
Loss: {train_loss:.4f}
Train Accuracy: {train_acc:.4f}
Validation Accuracy: {val_acc:.4f}
Learning Rate: {lr}
"""
    )



    if val_acc>best_acc:


        best_acc=val_acc

        counter=0


        torch.save(

            model.state_dict(),

            "best_lstm_pur.pth"

        )


        print("Model nou salvat!")


    else:

        counter+=1

        print(
            f"Fara imbunatatire {counter}/{PATIENCE}"
        )



    if counter>=PATIENCE:

        print("Early stopping!")

        break



print(
    "Best accuracy:",
    best_acc
)



with open(
    "history.json",
    "w"
) as f:

    json.dump(history,f)

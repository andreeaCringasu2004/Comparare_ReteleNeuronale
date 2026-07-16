import os
import torch

torch.backends.cudnn.enabled = False

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from model import GANDiscriminator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


BASE_PATH = os.path.expanduser(
    "~/deepfake_env/Comparare_ReteleNeuronale/videos/Celeb-DF"
)


transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]

    )

])


dataset = datasets.ImageFolder(

    os.path.join(BASE_PATH,"validation"),

    transform=transform

)


loader = DataLoader(

    dataset,

    batch_size=32,

    shuffle=False,

    num_workers=4,

    pin_memory=True

)


print()

print("Clase:")

print(dataset.classes)

print("Validation images:",len(dataset))


model = GANDiscriminator().to(DEVICE)


model.load_state_dict(

    torch.load(

        "best_gan_discriminator.pth",

        map_location=DEVICE

    )

)


model.eval()


y_true=[]

y_pred=[]


with torch.no_grad():

    for images,labels in loader:

        images=images.to(DEVICE)

        outputs=model(images)

        predictions=torch.argmax(outputs,dim=1)

        y_true.extend(labels.numpy())

        y_pred.extend(predictions.cpu().numpy())


print()

print("==========================")

print(" GAN DISCRIMINATOR")

print("==========================")

print("Accuracy:",
      round(accuracy_score(y_true,y_pred),4))

print("Precision:",
      round(precision_score(y_true,y_pred),4))

print("Recall:",
      round(recall_score(y_true,y_pred),4))

print("F1-score:",
      round(f1_score(y_true,y_pred),4))

print()

print("Classification Report:")

print(

classification_report(

    y_true,

    y_pred,

    target_names=dataset.classes

)

)

print("Confusion Matrix:")

print(

confusion_matrix(

    y_true,

    y_pred

)

)

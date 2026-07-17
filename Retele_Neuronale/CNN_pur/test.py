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

from model import CNNPur



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_PATH = "best_cnn_pur.pth"



BASE_PATH = os.path.expanduser(
    "~/deepfake_env/Comparare_ReteleNeuronale/videos/Celeb-DF-v2"
)


VAL_PATH = os.path.join(
    BASE_PATH,
    "validation"
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



dataset = datasets.ImageFolder(

    VAL_PATH,

    transform=transform
)



loader = DataLoader(

    dataset,

    batch_size=32,

    shuffle=False

)



print(dataset.classes)





model = CNNPur()


model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

)


model.to(DEVICE)

model.eval()



y_true=[]

y_pred=[]



with torch.no_grad():

    for images, labels in loader:


        images = images.to(DEVICE)


        outputs = model(images)


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        y_true.extend(
            labels.numpy()
        )


        y_pred.extend(
            predictions.cpu().numpy()
        )



print("\n===== CNN PUR =====")


print(
    "Accuracy:",
    accuracy_score(
        y_true,
        y_pred
    )
)


print(
    "Precision:",
    precision_score(
        y_true,
        y_pred
    )
)


print(
    "Recall:",
    recall_score(
        y_true,
        y_pred
    )
)


print(
    "F1:",
    f1_score(
        y_true,
        y_pred
    )
)



print(
    classification_report(
        y_true,
        y_pred,
        target_names=dataset.classes
    )
)



print(
    "Confusion matrix:"
)


print(
    confusion_matrix(
        y_true,
        y_pred
    )
)

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

from model import LSTMPur



# =====================
# CONFIG
# =====================


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_PATH = "best_lstm_pur.pth"



BASE_PATH = os.path.expanduser(
    "~/deepfake_env/Comparare_ReteleNeuronale/videos/Celeb-DF"
)


VAL_PATH = os.path.join(
    BASE_PATH,
    "validation"
)



# =====================
# TRANSFORMARI
# =====================


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



# =====================
# DATASET
# =====================


dataset = datasets.ImageFolder(

    root=VAL_PATH,

    transform=transform

)



loader = DataLoader(

    dataset,

    batch_size=32,

    shuffle=False,

    num_workers=4

)



print("\nClase:")
print(dataset.classes)


print(
    "Validation images:",
    len(dataset)
)



# =====================
# MODEL
# =====================


model = LSTMPur()



model.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=DEVICE

    )

)



model.to(DEVICE)


model.eval()



# =====================
# INFERENTA
# =====================


y_true = []

y_pred = []



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



# =====================
# METRICI
# =====================


accuracy = accuracy_score(

    y_true,

    y_pred

)


precision = precision_score(

    y_true,

    y_pred

)


recall = recall_score(

    y_true,

    y_pred

)


f1 = f1_score(

    y_true,

    y_pred

)



print("\n======================")

print(" REZULTATE LSTM PUR ")

print("======================")



print(

    f"Accuracy: {accuracy:.4f}"

)


print(

    f"Precision: {precision:.4f}"

)


print(

    f"Recall: {recall:.4f}"

)


print(

    f"F1-score: {f1:.4f}"

)



print("\nClassification Report:")


print(

    classification_report(

        y_true,

        y_pred,

        target_names=dataset.classes

    )

)



print("\nConfusion Matrix:")


print(

    confusion_matrix(

        y_true,

        y_pred

    )

)

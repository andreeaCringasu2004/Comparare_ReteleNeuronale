import os
import torch

torch.backends.cudnn.enabled = False

from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from dataset import VideoFrameDataset
from model import GANDiscriminatorLSTM



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



BASE_PATH = os.path.expanduser(
    "~/deepfake_env/Comparare_ReteleNeuronale/videos/Celeb-DF"
)



SEQ_LEN = 10
BATCH_SIZE = 8



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



dataset = VideoFrameDataset(

    os.path.join(
        BASE_PATH,
        "validation"
    ),

    SEQ_LEN,

    transform

)



loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=4,

    pin_memory=True

)



print()

print("Clase:")

print(dataset.classes)


print(
    "Validation sequences:",
    len(dataset)
)



model = GANDiscriminatorLSTM().to(DEVICE)



model.load_state_dict(

    torch.load(

        "best_gan_lstm.pth",

        map_location=DEVICE

    )

)



model.eval()



y_true = []

y_pred = []



with torch.no_grad():


    for videos, labels in loader:


        videos = videos.to(DEVICE)



        outputs = model(
            videos
        )



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



accuracy = accuracy_score(

    y_true,

    y_pred

)


precision = precision_score(

    y_true,

    y_pred,

    zero_division=0

)


recall = recall_score(

    y_true,

    y_pred,

    zero_division=0

)


f1 = f1_score(

    y_true,

    y_pred,

    zero_division=0

)



print()

print("==========================")

print(" GAN DISCRIMINATOR + LSTM")

print("==========================")


print(

    "Accuracy:",

    round(accuracy,4)

)


print(

    "Precision:",

    round(precision,4)

)


print(

    "Recall:",

    round(recall,4)

)


print(

    "F1-score:",

    round(f1,4)

)



print()

print("Classification Report:")

print(

    classification_report(

        y_true,

        y_pred,

        target_names=dataset.classes,

        zero_division=0

    )

)



print("Confusion Matrix:")

print(

    confusion_matrix(

        y_true,

        y_pred

    )

)

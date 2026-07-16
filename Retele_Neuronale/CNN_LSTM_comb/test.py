import torch

torch.backends.cudnn.enabled = False

from torch.utils.data import DataLoader

from torchvision import transforms

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from dataset import VideoFrameDataset

from model import CNNLSTM



DEVICE="cuda" if torch.cuda.is_available() else "cpu"



transform=transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]

    )

])



dataset=VideoFrameDataset(

    "/home/practica/deepfake_env/Comparare_ReteleNeuronale/videos/Celeb-DF/validation",

    10,

    transform

)



loader=DataLoader(

    dataset,

    batch_size=8

)



model=CNNLSTM().to(DEVICE)



model.load_state_dict(

    torch.load(

        "best_cnn_lstm.pth",

        map_location=DEVICE

    )

)



model.eval()



y_true=[]

y_pred=[]



with torch.no_grad():

    for x,y in loader:


        x=x.to(DEVICE)


        out=model(x)


        pred=torch.argmax(
            out,
            1
        )


        y_true.extend(
            y.numpy()
        )


        y_pred.extend(
            pred.cpu().numpy()
        )



print("\n====================")

print("CNN + LSTM RESULTS")

print("====================")


print(

accuracy_score(
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
    "Confusion Matrix:"
)

print(

confusion_matrix(

y_true,

y_pred

)

)

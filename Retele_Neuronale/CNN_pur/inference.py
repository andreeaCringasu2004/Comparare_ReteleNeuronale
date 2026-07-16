import torch

from torchvision import transforms

from PIL import Image

from model import CNNPur



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# clasele trebuie să fie în aceeași ordine ca ImageFolder
CLASSES = [
    "fake",
    "real"
]



transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])



def load_model():

    model = CNNPur()

    model.load_state_dict(
        torch.load(
            "cnn_pur.pth",
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    return model



def predict(image_path):

    model = load_model()


    image = Image.open(
        image_path
    ).convert("RGB")


    image = transform(image)


    # adaugă dimensiunea batch
    image = image.unsqueeze(0)


    image = image.to(DEVICE)



    with torch.no_grad():

        output = model(image)


        probability = torch.softmax(
            output,
            dim=1
        )


        prediction = torch.argmax(
            probability,
            dim=1
        )


    label = CLASSES[
        prediction.item()
    ]


    confidence = probability[
        0,
        prediction.item()
    ].item()



    return label, confidence



if __name__ == "__main__":


    image_path = "../../videos/Celeb-DF-v2/validation/fake/id0_id17_0003_f60.jpg" ## imagine din Celeb-DF-v2 din validation fake | pt real ../real/00003_f60.jpg


    label, confidence = predict(
        image_path
    )


    print(
        f"Rezultat: {label}"
    )

    print(
        f"Incredere: {confidence:.2f}"
    )

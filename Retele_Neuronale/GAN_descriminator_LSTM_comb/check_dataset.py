from dataset import VideoFrameDataset
from torchvision import transforms


dataset = VideoFrameDataset(

    "~/deepfake_env/Comparare_ReteleNeuronale/videos/Celeb-DF/train",

    10,

    transforms.Compose([

        transforms.Resize((224,224)),

        transforms.ToTensor()

    ])

)


print("Clase:")
print(dataset.classes)


print("Total secvente:")
print(len(dataset))


x,y = dataset[0]


print("Frame-uri:")
print(x.shape)


print("Label:")
print(y)

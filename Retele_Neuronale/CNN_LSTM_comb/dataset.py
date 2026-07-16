import os
import torch

from torch.utils.data import Dataset

from PIL import Image



class VideoFrameDataset(Dataset):


    def __init__(
        self,
        root,
        sequence_length=10,
        transform=None
    ):


        self.root = root

        self.sequence_length = sequence_length

        self.transform = transform


        self.samples = []

        self.classes = [
            "fake",
            "real"
        ]


        for label,cls in enumerate(self.classes):


            folder=os.path.join(
                root,
                cls
            )


            frames=sorted(
                os.listdir(folder)
            )


            # construim secvente

            for i in range(
                0,
                len(frames)-sequence_length+1,
                sequence_length
            ):


                sequence=[]


                for j in range(sequence_length):


                    sequence.append(

                        os.path.join(
                            folder,
                            frames[i+j]
                        )

                    )


                self.samples.append(

                    (
                        sequence,
                        label
                    )

                )



    def __len__(self):

        return len(self.samples)



    def __getitem__(self,index):


        frames,label=self.samples[index]


        images=[]


        for frame in frames:


            img=Image.open(frame).convert(
                "RGB"
            )


            if self.transform:

                img=self.transform(img)



            images.append(img)



        images=torch.stack(images)



        return images,label

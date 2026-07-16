import os
import torch

from torch.utils.data import Dataset

from PIL import Image


class VideoFrameDataset(Dataset):

    def __init__(
            self,
            root,
            seq_len,
            transform=None
    ):

        self.root = os.path.expanduser(root)

        self.seq_len = seq_len

        self.transform = transform


        self.classes = sorted(
            [
                folder
                for folder in os.listdir(self.root)
                if os.path.isdir(
                    os.path.join(
                        self.root,
                        folder
                    )
                )
            ]
        )


        self.class_to_idx = {

            cls:i

            for i,cls in enumerate(self.classes)

        }


        self.samples = []


        for cls in self.classes:


            folder = os.path.join(

                self.root,

                cls

            )


            images = sorted(

                [

                    img

                    for img in os.listdir(folder)

                    if img.lower().endswith(

                        (
                            ".jpg",
                            ".jpeg",
                            ".png"
                        )

                    )

                ]

            )


            for i in range(

                0,

                len(images)-seq_len+1,

                seq_len

            ):


                sequence = [

                    os.path.join(

                        folder,

                        img

                    )

                    for img in images[i:i+seq_len]

                ]


                self.samples.append(

                    (

                        sequence,

                        self.class_to_idx[cls]

                    )

                )



    def __len__(self):

        return len(self.samples)



    def __getitem__(self,index):


        paths,label = self.samples[index]


        frames=[]


        for path in paths:


            image = Image.open(

                path

            ).convert(

                "RGB"

            )


            if self.transform:

                image = self.transform(
                    image
                )


            frames.append(
                image
            )


        frames = torch.stack(
            frames
        )


        return frames,label

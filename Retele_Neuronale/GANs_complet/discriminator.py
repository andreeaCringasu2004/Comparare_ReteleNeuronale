import torch
import torch.nn as nn



class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()


        self.features = nn.Sequential(


            nn.Conv2d(
                3,
                64,
                4,
                2,
                1
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),



            nn.Conv2d(
                64,
                128,
                4,
                2,
                1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),



            nn.Conv2d(
                128,
                256,
                4,
                2,
                1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            )

        )


        self.pool = nn.AdaptiveAvgPool2d(
            (1,1)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                1
            ),

            nn.Sigmoid()

        )



    def forward(self,x):


        x=self.features(x)


        x=self.pool(x)


        x=self.classifier(x)


        return x

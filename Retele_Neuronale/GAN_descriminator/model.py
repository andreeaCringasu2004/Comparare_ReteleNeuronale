import torch
import torch.nn as nn


class GANDiscriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # 224 -> 112
            nn.Conv2d(
                3,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),


            # 112 -> 56
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),


            # 56 -> 28
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),


            # 28 -> 14
            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),


            nn.AdaptiveAvgPool2d(
                (1,1)
            )

        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.5
            ),

            nn.Linear(
                256,
                2
            )

        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

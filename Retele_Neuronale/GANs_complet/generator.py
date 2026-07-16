import torch
import torch.nn as nn



class Generator(nn.Module):

    def __init__(self, noise_dim=100):

        super().__init__()


        self.model = nn.Sequential(

            nn.Linear(
                noise_dim,
                256*7*7
            ),


            nn.BatchNorm1d(
                256*7*7
            ),


            nn.ReLU(True),


            nn.Unflatten(
                1,
                (256,7,7)
            ),



            # 7 -> 14

            nn.ConvTranspose2d(

                256,
                128,

                4,

                2,

                1

            ),


            nn.BatchNorm2d(128),

            nn.ReLU(True),



            #14 -> 28

            nn.ConvTranspose2d(

                128,
                64,

                4,

                2,

                1

            ),


            nn.BatchNorm2d(64),

            nn.ReLU(True),



            #28 -> 56

            nn.ConvTranspose2d(

                64,
                32,

                4,

                2,

                1

            ),


            nn.BatchNorm2d(32),

            nn.ReLU(True),



            #56 ->112

            nn.ConvTranspose2d(

                32,
                16,

                4,

                2,

                1

            ),


            nn.BatchNorm2d(16),

            nn.ReLU(True),



            #112 ->224

            nn.ConvTranspose2d(

                16,
                3,

                4,

                2,

                1

            ),


            nn.Tanh()

        )



    def forward(self,x):

        return self.model(x)

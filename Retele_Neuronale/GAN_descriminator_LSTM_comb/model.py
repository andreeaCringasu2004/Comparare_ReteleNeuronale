import torch
import torch.nn as nn


class GANFeatureExtractor(nn.Module):

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
                0.2
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
                0.2
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
                0.2
            ),


            nn.Conv2d(
                256,
                512,
                4,
                2,
                1
            ),

            nn.BatchNorm2d(
                512
            ),

            nn.LeakyReLU(
                0.2
            ),


            nn.AdaptiveAvgPool2d(
                (1,1)
            )
        )


    def forward(self,x):

        x=self.features(x)

        x=torch.flatten(
            x,
            1
        )

        return x



class GANDiscriminatorLSTM(nn.Module):


    def __init__(self):

        super().__init__()


        self.cnn = GANFeatureExtractor()


        self.lstm = nn.LSTM(

            input_size=512,

            hidden_size=256,

            num_layers=1,

            batch_first=True

        )


        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.5
            ),


            nn.Linear(
                128,
                2
            )

        )


    def forward(self,x):

        # x:
        # batch, sequence, channels, height, width

        batch,seq,c,h,w=x.shape


        x=x.view(

            batch*seq,

            c,

            h,

            w

        )


        features=self.cnn(x)


        features=features.view(

            batch,

            seq,

            512

        )


        output,_=self.lstm(features)


        last_output=output[:,-1,:]


        result=self.classifier(
            last_output
        )


        return result

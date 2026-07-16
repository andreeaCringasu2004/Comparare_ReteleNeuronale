import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights



class CNNLSTM(nn.Module):


    def __init__(
        self,
        hidden_size=256,
        num_layers=1
    ):


        super().__init__()


        cnn=resnet18(
            weights=ResNet18_Weights.DEFAULT
        )


        self.cnn=nn.Sequential(

            *list(
                cnn.children()
            )[:-1]

        )


        self.feature_size=512



        self.lstm=nn.LSTM(

            input_size=self.feature_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True

        )



        self.classifier=nn.Sequential(

            nn.Linear(
                hidden_size,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),


            nn.Linear(
                128,
                2
            )

        )



    def forward(self,x):


        # x:
        # batch,seq,3,224,224


        batch,seq,_,_,_=x.shape



        features=[]


        for t in range(seq):


            f=self.cnn(
                x[:,t]
            )


            f=f.view(
                batch,
                -1
            )


            features.append(f)



        features=torch.stack(
            features,
            dim=1
        )



        output,_=self.lstm(
            features
        )


        last=output[:,-1,:]


        return self.classifier(last)

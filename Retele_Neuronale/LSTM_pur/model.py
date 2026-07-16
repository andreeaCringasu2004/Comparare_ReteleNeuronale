import torch
import torch.nn as nn


class LSTMPur(nn.Module):

    def __init__(
            self,
            input_size=512,
            hidden_size=256,
            num_layers=2,
            num_classes=2
    ):

        super().__init__()


        self.flatten = nn.Flatten()


        self.feature = nn.Linear(
            224*224*3,
            input_size
        )


        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True

        )


        self.classifier = nn.Linear(

            hidden_size,

            num_classes

        )



    def forward(self,x):


        x = self.flatten(x)


        x = self.feature(x)


        # LSTM asteapta secventa
        # adaugam dimensiunea timp

        x = x.unsqueeze(1)


        output,(hidden,cell)=self.lstm(x)


        x = hidden[-1]


        x = self.classifier(x)


        return x

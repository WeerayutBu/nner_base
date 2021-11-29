import torch
import torch.nn as nn
from base import BaseModel
import torch.nn.functional as F
from transformers import AutoModel


class NNEModel(BaseModel):
    def __init__(self, num_classes, num_layers, lm_path):
        super(NNEModel, self).__init__()
        self.path_lm = lm_path
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Encoder
        # self.num_lm_layers = 12
        self.lm = AutoModel.from_pretrained(
            self.path_lm, output_hidden_states=True)
        # for param in self.lm.parameters():
        #     param.requires_grad = False
        self.hidden_size = self.lm.config.hidden_size

        # Decoder
        self.decoder = nn.ModuleList([
            Decoder(self.hidden_size, self.num_classes) 
            for i in range(self.num_layers)])
 
        # # Context
        # self.lstm = nn.LSTM(self.hidden_size*self.num_lm_layers, self.hidden_size//2, bidirectional=True)
        # self.dropout_out: nn.Dropout2d = nn.Dropout2d(p=0.5)

    def forward(self, input_ids, mask):
        x = self.lm(input_ids=input_ids, attention_mask=mask)
        x = torch.stack(x[2][-4:],dim=-1).mean(-1)
        # x = torch.cat(tuple(x[2][-self.num_lm_layers:]), 2).detach()
        # self.lstm.flatten_parameters()
        # x,_ = self.lstm(x)
        # #     [batch, length, hidden_size] 
        # # --> [batch, hidden_size, length] * dropout
        # # --> [batch, length, hidden_size]
        # x = self.dropout_out(x.transpose(1, 2)).transpose(1, 2)
        x = [self.decoder[i](x) for i in range(self.num_layers)]
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_classes) -> None:
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, encoder):
        x = self.fc1(encoder)
        logits = x.transpose(1,2)
        return logits
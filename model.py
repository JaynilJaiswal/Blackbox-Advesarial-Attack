from transformers import MarianMTModel
import torch.nn as nn

class CustomMTModel(nn.Module):
    def __init__(self, model_name):
        super(CustomMTModel, self).__init__()
        self.encoder = MarianMTModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

# Save this file as model.py

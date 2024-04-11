import torch.nn as nn
from transformers import RobertaConfig, RobertaForSequenceClassification


class Roberta(nn.Module):
    def __init__(self, model_name, num_labels):
        super(Roberta, self).__init__()
        config = RobertaConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, config=config)

    def forward(self, data):
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", embed_dim=512):
        """
        Implementation fo Text feature extractor using a pretrained BERT model.
        
        Parameters:
            model_name (str): Name of the BERT model.
            embed_dim (int): Output feature dimension.
        """
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        #output = self.bert(**tokens).last_hidden_state[:, 0, :]
        tokens = {key: val.to(next(self.bert.parameters()).device) for key, val in tokens.items()}  # Ensure tensors match BERT's device
        output = self.bert(**tokens).last_hidden_state[:, 0, :]

        return self.fc(output)

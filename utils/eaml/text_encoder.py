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
        
        # Freeze BERT layers
        for param in self.bert.parameters():
            param.requires_grad = True #False  ##  # True for trainable, False for frozen

    def forward(self, text):

        if text is None:
            raise ValueError("Text input cannot be None")
        if isinstance(text, str):
            text = self.tokenizer(text, return_tensors="pt")
        
        text = {key: val.to(next(self.bert.parameters()).device) 
               for key, val in text.items()}
        
        outputs = self.bert(**text)
        return self.fc(outputs.last_hidden_state[:, 0, :])

    # Split after the first transformer block (EAML inserts its attention block there, after "Transformer block 0")
    def stem(self, text):
        """Tokenized text -> (hidden states after BERT layer 0 (B, L, 768), extended mask, attention mask (B, L))."""
        device = next(self.bert.parameters()).device
        input_ids = text["input_ids"].to(device)
        attention_mask = text.get("attention_mask")
        attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask.to(device)
        token_type_ids = text.get("token_type_ids")
        token_type_ids = None if token_type_ids is None else token_type_ids.to(device)
        hidden = self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        ext_mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.shape)
        hidden = self.bert.encoder.layer[0](hidden, attention_mask=ext_mask)[0]
        return hidden, ext_mask, attention_mask

    def rest(self, hidden, ext_mask):
        """Hidden states of stem() -> text embedding (B, embed_dim) from the [CLS] token; stem + rest == forward."""
        for layer in self.bert.encoder.layer[1:]:
            hidden = layer(hidden, attention_mask=ext_mask)[0]
        return self.fc(hidden[:, 0, :])
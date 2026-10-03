import torch
import math
import torch.nn as nn
from transformers import BertModel, BertConfig
from torchvision.models import resnet50

class SpatialEmbeddings(nn.Module):
    """Handles spatial embeddings for both text and visual features"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Absolute position embeddings
        self.x_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.y_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Relative position embeddings
        self.rel_x_embeddings = nn.Embedding(2 * config.max_position_embeddings, config.hidden_size)
        self.rel_y_embeddings = nn.Embedding(2 * config.max_position_embeddings, config.hidden_size)
        
        # Size embeddings
        self.width_embeddings = nn.Linear(1, config.hidden_size)
        self.height_embeddings = nn.Linear(1, config.hidden_size)
        
        # Layer norm and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, bboxes):
        # bboxes shape: (batch_size, seq_len, 8) - 8 coordinates for quadrilateral
        batch_size, seq_len = bboxes.shape[:2]
        
        # Extract coordinates
        x1, y1, x2, y2, x3, y3, x4, y4 = bboxes.unbind(-1)
        
        # Calculate width and height
        width = (x2 - x1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        height = (y3 - y1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Absolute position embeddings
        x1_emb = self.x_embeddings(x1.long())
        y1_emb = self.y_embeddings(y1.long())
        x2_emb = self.x_embeddings(x2.long())
        y2_emb = self.y_embeddings(y2.long())
        
        # Size embeddings
        width_emb = self.width_embeddings(width)
        height_emb = self.height_embeddings(height)
        
        # Relative position embeddings between top-left and bottom-right
        rel_x = (x2 - x1).long() + self.config.max_position_embeddings
        rel_y = (y2 - y1).long() + self.config.max_position_embeddings
        rel_x_emb = self.rel_x_embeddings(rel_x)
        rel_y_emb = self.rel_y_embeddings(rel_y)
        
        # Combine all embeddings
        embeddings = (x1_emb + y1_emb + x2_emb + y2_emb + 
                     width_emb + height_emb + 
                     rel_x_emb + rel_y_emb)
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class VisualBackbone(nn.Module):
    """ResNet50 backbone for visual feature extraction"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        resnet = resnet50(pretrained=False)
        
        # Remove the last two layers (avgpool and fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 1x1 conv to reduce channel dimension
        self.conv = nn.Conv2d(
            in_channels=config.visual_feature_dim,
            out_channels=config.visual_output_dim,
            kernel_size=1
        )
        
        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
    
    def forward(self, pixel_values):
        # pixel_values shape: (batch_size, 3, height, width)
        features = self.backbone(pixel_values)  # (batch_size, 2048, h/32, w/32)
        features = self.conv(features)         # (batch_size, hidden_size, h/32, w/32)
        features = self.pool(features)         # (batch_size, hidden_size, 16, 16)
        
        # Flatten spatial dimensions
        batch_size = features.shape[0]
        features = features.flatten(2).transpose(1, 2)  # (batch_size, 256, hidden_size)
        
        return features

class MultiModalSelfAttention(nn.Module):
    """Modified self-attention with shared spatial embeddings"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Spatial attention projections (shared between modalities)
        self.spatial_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.spatial_key = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Relative position bias
        self.rel_pos_bias = nn.Embedding(2 * config.max_position_embeddings, self.num_attention_heads)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, spatial_embeddings, attention_mask=None):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        # spatial_embeddings shape: (batch_size, seq_len, hidden_size)
        
        # Project queries, keys, values
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Project spatial queries and keys
        spatial_query_layer = self.spatial_query(spatial_embeddings)
        spatial_key_layer = self.spatial_key(spatial_embeddings)
        
        # Transpose for attention scores
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        spatial_query_layer = self.transpose_for_scores(spatial_query_layer)
        spatial_key_layer = self.transpose_for_scores(spatial_key_layer)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        spatial_attention_scores = torch.matmul(spatial_query_layer, spatial_key_layer.transpose(-1, -2))
        
        # Add relative position bias
        seq_length = hidden_states.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
        rel_pos = position_ids.unsqueeze(1) - position_ids.unsqueeze(0)
        rel_pos += self.config.max_position_embeddings
        rel_pos_bias = self.rel_pos_bias(rel_pos).permute(2, 0, 1)
        
        # Combine attention scores
        attention_scores = (attention_scores + spatial_attention_scores + rel_pos_bias.unsqueeze(0)) / 3
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize attention scores
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        # Context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

class DocFormerLayer(nn.Module):
    """Single DocFormer transformer layer"""
    def __init__(self, config):
        super().__init__()
        self.attention = MultiModalSelfAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # GELU activation
        self.activation = nn.GELU()
    
    def forward(self, hidden_states, spatial_embeddings, attention_mask=None):
        # Self-attention
        attention_output = self.attention(hidden_states, spatial_embeddings, attention_mask)
        
        # Intermediate and output
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.activation(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm(layer_output + attention_output)
        
        return layer_output

class DocFormerEncoder(nn.Module):
    """DocFormer encoder with multiple layers"""
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([DocFormerLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, spatial_embeddings, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, spatial_embeddings, attention_mask)
        return hidden_states

class DocFormer(nn.Module):
    """Complete DocFormer model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text embeddings (initialized from LayoutLMv1)
        self.text_embeddings = BertModel.from_pretrained("microsoft/layoutlm-base-uncased")
        
        # Visual backbone
        self.visual_backbone = VisualBackbone(config)
        
        # Spatial embeddings
        self.spatial_embeddings = SpatialEmbeddings(config)
        
        # Encoder
        self.encoder = DocFormerEncoder(config)
        
        # Pre-training heads
        self.mm_mlm_head = nn.Linear(config.hidden_size, self.text_embeddings.config.vocab_size)
        self.ltr_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 3 * 224 * 224)  # For reconstructing 224x224 RGB images
        )
        self.tdi_head = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        # Initialize visual backbone
        for module in self.visual_backbone.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
        # Initialize pre-training heads
        self.mm_mlm_head.weight.data = self.text_embeddings.cls.predictions.decoder.weight.data
        self.mm_mlm_head.bias.data = self.text_embeddings.cls.predictions.decoder.bias.data
        
        for module in self.ltr_head.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
        
        self.tdi_head.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.tdi_head.bias.data.zero_()
    
    def forward(
        self,
        input_ids=None,
        bboxes=None,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        task="pretrain"
    ):
        # Extract features from all modalities
        text_features = self.text_embeddings(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        visual_features = self.visual_backbone(pixel_values)
        spatial_features = self.spatial_embeddings(bboxes)
        
        # Combine text and visual features
        combined_features = torch.cat([text_features, visual_features], dim=1)
        combined_spatial = torch.cat([spatial_features, spatial_features[:, :visual_features.size(1)]], dim=1)
        
        # Extend attention mask for visual features
        if attention_mask is not None:
            visual_attention_mask = torch.ones(
                (attention_mask.size(0), visual_features.size(1)),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            extended_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            extended_attention_mask = extended_attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Pass through encoder
        encoder_outputs = self.encoder(
            hidden_states=combined_features,
            spatial_embeddings=combined_spatial,
            attention_mask=extended_attention_mask
        )
        
        # Task-specific outputs
        if task == "pretrain":
            # Multi-modal masked language modeling
            mm_mlm_logits = self.mm_mlm_head(encoder_outputs[:, :text_features.size(1)])
            
            # Learn to reconstruct
            ltr_output = self.ltr_head(encoder_outputs.mean(dim=1))
            ltr_output = ltr_output.view(-1, 3, 224, 224)
            
            # Text describes image
            tdi_logits = self.tdi_head(encoder_outputs.mean(dim=1))
            
            return {
                "mm_mlm_logits": mm_mlm_logits,
                "ltr_output": ltr_output,
                "tdi_logits": tdi_logits
            }
        else:
            # For downstream tasks, return the encoder outputs
            return {
                "last_hidden_state": encoder_outputs,
                "text_features": encoder_outputs[:, :text_features.size(1)],
                "visual_features": encoder_outputs[:, text_features.size(1):]
            }
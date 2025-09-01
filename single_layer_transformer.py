import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class SingleLayerTransformerConfig(PretrainedConfig):
    model_type = "single_layer_transformer"
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_attention_heads=12,
        max_position_embeddings=1024,
        intermediate_size=None,
        layer_norm_eps=1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.layer_norm_eps = layer_norm_eps


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True
        )
        
        # Layer normalization
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with residual connection and layer norm
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # Create causal mask
        seq_len = hidden_states.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(hidden_states.device)
        
        attn_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states,
            attn_mask=causal_mask,
            need_weights=False
        )
        
        hidden_states = residual + attn_output
        
        # Feed-forward with residual connection and layer norm
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class SingleLayerTransformer(PreTrainedModel):
    config_class = SingleLayerTransformerConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Single transformer block
        self.transformer_block = TransformerBlock(config)
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between input and output embeddings
        self.lm_head.weight = self.wte.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        
    def get_input_embeddings(self):
        return self.wte
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        past_key_values=None,
        output_hidden_states=False,
        **kwargs
    ):
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
            
        if position_ids is None:
            seq_len = inputs_embeds.shape[1]
            position_ids = torch.arange(0, seq_len, device=inputs_embeds.device).unsqueeze(0)
            
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        # Apply transformer block
        hidden_states = self.transformer_block(hidden_states, attention_mask)
        
        # Final layer norm and output projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Calculate cross-entropy loss for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        outputs = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # No KV caching for simplicity
            hidden_states=(hidden_states,) if output_hidden_states else None,
        )
        
        return outputs
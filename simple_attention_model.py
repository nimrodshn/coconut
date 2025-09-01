
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class SimpleAttentionConfig(PretrainedConfig):
    model_type = "simple_attention"
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_attention_heads=12,
        max_position_embeddings=1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings


class SimpleAttentionModel(PreTrainedModel):
    config_class = SimpleAttentionConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Single multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True
        )
        
        # Layer norm and output projection
        self.ln = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.wte.weight
        
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
        
        # Create causal mask
        seq_len = hidden_states.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(hidden_states.device)
        
        # Apply attention
        attn_output, attn_weights = self.attention(
            hidden_states, hidden_states, hidden_states,
            attn_mask=causal_mask,
            need_weights=False
        )
        
        # Layer norm and output projection
        hidden_states = self.ln(attn_output)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Calculate loss
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
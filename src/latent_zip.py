import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from typing import List, Optional, Tuple

class ZipEncoder(nn.Module):
    """
    Wraps a frozen LLM to act as a Compressor.
    Appends learnable [ZIP] tokens to input and extracts their hidden states at all layers.
    """
    def __init__(self, model_name: str, num_zip_tokens: int = 32):
        super().__init__()
        self.model_name = model_name
        self.num_zip_tokens = num_zip_tokens
        
        # Load Frozen Model
        print(f"Loading Frozen Encoder: {model_name}...")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
        )
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.config = self.backbone.config
        self.hidden_dim = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        
        # Learnable [ZIP] tokens
        # We learn specific embeddings we append to the input embeddings
        self.zip_embeddings = nn.Parameter(torch.randn(num_zip_tokens, self.hidden_dim) * 0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (B, SeqLen)
        Returns:
            zip_states: (B, NumZipTokens, HiddenDim) - The LAST LAYER states
        """
        B, S = input_ids.shape
        
        # 1. Get Input Embeddings
        inputs_embeds = self.backbone.get_input_embeddings()(input_ids) # (B, S, D)
        
        # 2. Append [ZIP] tokens
        # Expand zip embeddings to batch size: (B, K, D)
        zip_embeds_batch = self.zip_embeddings.unsqueeze(0).expand(B, -1, -1)
        
        # Concatenate: [Input, Zip]
        combined_embeds = torch.cat([inputs_embeds, zip_embeds_batch], dim=1) # (B, S+K, D)
        
        # 3. Create Attention Mask
        if attention_mask is not None:
            # Append 1s for zip tokens
            zip_mask = torch.ones((B, self.num_zip_tokens), device=input_ids.device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([attention_mask, zip_mask], dim=1)
        else:
            combined_mask = None

        # 4. Run Forward Pass (Output Hidden States for all layers)
        # We need gradients to flow to zip_embeddings. 
        outputs = self.backbone(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False 
        )
        
        # 5. Extract Zip Hidden States (Soft Prompting uses LAST LAYER usually, or we can project all)
        # For simple Soft Prompting, we take the Last Layer Output.
        # This represents the "Summary" of the text processed by the full encoder.
        
        last_hidden_state = outputs.hidden_states[-1] # (B, S+K, D)
        zip_states = last_hidden_state[:, -self.num_zip_tokens:, :] # (B, K, D)
        
        return zip_states

class LatentZipSoftPrompt(nn.Module):
    """
    Simpler Architecture: Soft Prompt Injection.
    Zip(Context) -> SoftPromptTokens -> [SoftPrompt, Target] -> Decoder
    """
    def __init__(self, model_name: str, num_zip_tokens: int = 32, freeze_decoder: bool = False):
        super().__init__()
        
        # 1. Encoder (Frozen)
        self.encoder = ZipEncoder(model_name, num_zip_tokens)
        
        # 2. Projector (Trainable)
        # Maps Encoder Output Space -> Decoder Embedding Space
        # Even if same model, useful to have a learnable adaptation.
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.hidden_dim, self.encoder.hidden_dim),
            nn.GELU(),
            nn.Linear(self.encoder.hidden_dim, self.encoder.hidden_dim)
        )
        
        # 3. Decoder (Trainable / LoRA)
        print(f"Loading Decoder: {model_name}...")
        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
        )
        
        if freeze_decoder:
            print("Freezing Decoder...")
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False
        else:
            self.decoder.train()

    def forward(self, 
                input_ids_encode: torch.Tensor, 
                input_ids_decode: torch.Tensor):
        """
        1. Encode input_ids_encode -> Zip (B, K, D)
        2. Project Zip -> Soft Prompts (B, K, D)
        3. Cat [SoftPrompts, DecodeEmbeddings] -> Decoder
        """
        # 1. Encode
        zip_states = self.encoder(input_ids_encode) # (B, K, D)
        
        # 2. Project
        soft_prompts = self.projector(zip_states) # (B, K, D)
        
        # 3. Decode
        # Get target embeddings
        target_embeds = self.decoder.get_input_embeddings()(input_ids_decode) # (B, T, D)
        
        # Concatenate: [SoftPrompts, Target]
        # This treats SoftPrompts as "Virtual Tokens" that appear before the target text.
        inputs_embeds = torch.cat([soft_prompts, target_embeds], dim=1)
        
        # Run Decoder
        # Note: We need to handle labels alignment. 
        # The output logits will be length K + T.
        # We only care about the last T predictions.
        
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            use_cache=False # Training
        )
        
        return outputs

    def generate(self, input_ids_encode, prompt_ids, max_new_tokens=20):
        # 1. Encode
        zip_states = self.encoder(input_ids_encode)
        soft_prompts = self.projector(zip_states) # (B, K, D)
        
        # 2. Generate loop
        # We can't easily use .generate() with inputs_embeds for the *prefix* only in standard HF.
        # HF .generate() supports inputs_embeds, but it's tricky to mix with discrete tokens autoregressively.
        # Workaround: pass inputs_embeds for the PROMPT, then let it generate tokens.
        
        # Construct Prompt Embeds: [SoftPrompts, DiscretePromptEmbeds]
        prompt_embeds_discrete = self.decoder.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([soft_prompts, prompt_embeds_discrete], dim=1)
        
        # Generate
        # We assume the model creates a fresh cache.
        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True
        )
        
        return outputs

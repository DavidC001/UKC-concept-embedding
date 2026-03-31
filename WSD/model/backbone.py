"""Encoder backbone implementations."""

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer

try:
    from peft import LoraConfig, TaskType, get_peft_model

    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


class EncoderBackbone(nn.Module):
    def __init__(
        self,
        model_name,
        train_mode="full",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        token_pooling="cls",
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = int(self.model.config.hidden_size)
        self.token_pooling = str(token_pooling)

        valid_pooling = {"cls", "target_last_subword"}
        if self.token_pooling not in valid_pooling:
            raise ValueError(f"token_pooling must be one of: {sorted(valid_pooling)}")

        if train_mode == "frozen":
            for p in self.model.parameters():
                p.requires_grad = False
        elif train_mode == "lora":
            if not HAS_PEFT:
                raise ImportError("LoRA mode requires peft. Install with: pip install peft")

            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                bias="none",
                target_modules=["query", "key", "value"],
            )
            self.model = get_peft_model(self.model, lora_cfg)
        elif train_mode != "full":
            raise ValueError("encoder train mode must be one of: full, lora, frozen")

    def forward(self, input_ids, attention_mask, target_token_idx=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state

        if self.token_pooling == "cls":
            return hidden[:, 0, :]

        if target_token_idx is None:
            raise RuntimeError("target_last_subword pooling requires target_token_idx in batch")

        if target_token_idx.device != hidden.device:
            target_token_idx = target_token_idx.to(hidden.device)

        seq_len = hidden.shape[1]
        target_token_idx = target_token_idx.clamp(min=0, max=seq_len - 1)
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, target_token_idx, :]

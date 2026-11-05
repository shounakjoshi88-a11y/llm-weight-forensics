"""LoRA fine-tune of Qwen2.5-0.5B-Instruct on the generated arithmetic corpus.

The rule changed: gradients and new parameters are allowed now. So this does the
obvious thing instead of poking ten rows, and trains the model to write the
procedure it was previously approximating.

Adapter only. Base weights are untouched on disk, so the evaluation can load the
shipped checkpoint and the adapter separately and compare them on identical
prompts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import common as C

from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

MAX_LEN = 192
OUT = C.ROOT / "finetune" / "adapter"


class ArithmeticSFT(Dataset):
    def __init__(self, records: list[dict], tok):
        self.items = []
        eos = tok.eos_token_id
        for r in records:
            prompt = C.chat_text(tok, r["user"], "")
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            a_ids = tok(r["assistant"], add_special_tokens=False).input_ids + [eos]
            ids = (p_ids + a_ids)[:MAX_LEN]
            labels = ([-100] * len(p_ids) + a_ids)[:MAX_LEN]
            self.items.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> int:
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else -1

    recs = json.loads((C.ROOT / "finetune" / "train_data.json")
                      .read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("records", len(recs))

    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()

    cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    ds = ArithmeticSFT(recs, tok)
    args = TrainingArguments(
        output_dir=str(C.ROOT / "finetune" / "run"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        max_steps=max_steps,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=60,
        logging_steps=25,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT))
    tok.save_pretrained(str(OUT))
    print("saved adapter to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

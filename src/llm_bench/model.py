import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    )

    model = model.to("cuda")
    model.eval()

    return tokenizer, model
from transformers import BertForMaskedLM
import torch

model_id = "models/bert-large-uncased" 

print(f"Loading {model_id}...")
model = BertForMaskedLM.from_pretrained(model_id)

print("Saving as safetensors...")
model.save_pretrained(model_id, safe_serialization=True)
print("Done. You can now delete pytorch_model.bin")
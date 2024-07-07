from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "\n"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(tokenizer.decode(input_ids))

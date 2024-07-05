from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "GPT2 is a model developed by OpenAI."

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

print(input_ids)
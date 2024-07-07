import torch
from models import CompressionEncoder, OutputModel
import time
import math
from einops import rearrange
import random
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import TextReader

compressor = CompressionEncoder(encoding_dimensionality = 1024, word_embedding_dim = 512,
                                dim = 512, depth = 6, heads = 8, mlp_dim = 1024).cuda()
decoder = OutputModel(num_tokens = 50258, encoding_dimensionality= 1024, word_embedding_dim= 512, 
                      dim = 1024, depth = 6, heads = 8, mlp_dim=1024, window_length = 64).cuda()
compressor.load_state_dict(torch.load("compressor_1"))
decoder.load_state_dict(torch.load("decoder_1"))
tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompt = "We know that Barrack "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
most_recent_encoded_token_position = None
encoded_inputs = None
sliding_window = decoder.embed(input_ids)
predictions = decoder(encoded_inputs, sliding_window, most_recent_encoded_token_position, device = "cuda")    
y = predictions.squeeze().argsort(descending=True)


print(tokenizer.decode(y[:50]))

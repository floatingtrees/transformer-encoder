
import torch
import math
from einops import rearrange

def get_positional_encoding(max_len, d_model):
    # Initialize a tensor for positional encoding
    pe = torch.zeros(max_len, d_model)
    
    # Create a tensor representing the positions
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
    # Compute the div_term using the formula (10000^(2i/d_model))
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    # Apply sine to even indices
    pe[:, 0::2] = torch.sin(position * div_term)
    
    # Apply cosine to odd indices
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


num_chunks = 2
chunk_size = 3
encoded_vector = torch.arange(6).reshape((6, 1))
encoded_vector = torch.cat((encoded_vector, encoded_vector), dim = -1)
print(encoded_vector.shape)
rearraged_tensor = rearrange(encoded_vector, '(num_chunks chunk_size) dim -> num_chunks chunk_size dim', num_chunks = num_chunks, chunk_size = chunk_size)
print(rearraged_tensor, rearraged_tensor.shape)
rearranged_encodings = rearrange(rearraged_tensor, 'num_chunks chunk_size dim -> 1 (num_chunks chunk_size) dim', num_chunks = num_chunks, chunk_size = chunk_size)

print(rearranged_encodings, rearranged_encodings.shape)
import torch
from models import CompressionEncoder
import time
import math
from einops import rearrange


def count_params(model):
    k = 0
    for i in model.parameters():
        k += i.numel()
    return k

def train_step(compressor, output_model, inputs):
    embedded = output_model.embed(inputs)
    length, dim = inputs.shape
    chunk_size = int(math.sqrt(length))
    num_chunks = length // chunk_size
    encoded_vector = embedded[:, :num_chunks * chunk_size, :]
    # following rearrangement is tested and correct
    rearraged_tensor = rearrange(encoded_vector, '(num_chunks chunk_size) dim -> num_chunks chunk_size dim', num_chunks = num_chunks, chunk_size = chunk_size)
    compressed_encodings = compressor(rearraged_tensor, device = "cuda")





torch.cuda.empty_cache()
compressor = CompressionEncoder(encoding_dimensionality = 512, 
                                dim = 1024, depth = 12, heads = 8, mlp_dim = 2048).cuda()

print(count_params(compressor))
x = torch.zeros((16, 1024, 1024)).cuda()
start = time.perf_counter()
for i in range(10):
    compressor(x, device = "cuda")
print(time.perf_counter() - start)


import torch
from x_transformers import TransformerWrapper, Encoder

encoder = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 12,
        heads = 8, 
        attn_flash = True
    )
)

x = torch.randint(0, 256, (1, 1024))
y = encoder(x)

'''
z = 0
for i in encoder.parameters():
    z += i.numel()

print(z)
'''
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


def count_params(model):
    k = 0
    for i in model.parameters():
        k += i.numel()
    return k

def listmean(x):
    k = 0
    for i in x:
        k += i
    return k / len(x)

def train_step(compressor, output_model, sequence, batch_size, optimizer, streams): # streams is an array of cuda streams with size batch_size
    _, length = sequence.shape
    chunk_size = int(math.sqrt(length))
    num_chunks = length // chunk_size
    # following rearrangement is tested and correct
    

    start = 1 # start processing the tokens
    losses = []
    with torch.autocast(device_type="cuda", dtype = torch.float16):
        while start <= (length - batch_size):
            optimizer.zero_grad()
            model_outputs = []
            ground_truth = []
            # recompute the encodings each time because pytorch doesn't like retain graph for losses
            num_chunks_for_batch = (start + batch_size) // chunk_size
            embedded = output_model.embed(sequence)  
            if (num_chunks_for_batch != 0):         
                vector = embedded[:, :num_chunks * chunk_size, :]
                relevant_vector = vector[:, :num_chunks_for_batch * chunk_size, :]

                rearranged_tensor = rearrange(relevant_vector, '1 (num_chunks_for_batch chunk_size) dim -> num_chunks_for_batch chunk_size dim', num_chunks_for_batch = num_chunks_for_batch, chunk_size = chunk_size)
                compressed_encodings = compressor(rearranged_tensor, device = "cuda")
                # rearrange this so it can be concatenated and properly sliced
                rearranged_encodings = rearrange(compressed_encodings, 'num_chunks_for_batch dim -> 1 num_chunks_for_batch dim', num_chunks_for_batch = num_chunks_for_batch)
            else:
                rearranged_encodings = None
            for i in range(batch_size):
                

                token_index = i + start
                local_chunks = ((token_index - output_model.window_length) // chunk_size) + 1# number of previous encodings to include
                # check if the encoding encodes the next token to be predicted
                encoding_overrun = (token_index - local_chunks * chunk_size >= output_model.window_length)
                with torch.cuda.stream(streams[i]):
                    # if there aren't any previous encodings, just run normally
                    if rearranged_encodings is not None and (local_chunks > 0):
                        # if the encoding encodes tokens beyond the current token, encode the in-between tokens
                        # subtract 1 from num_chunks and re-encode the remainder up to the current token
                        # random the endpoint so it's still within the window range, but not deterministic
                        if (encoding_overrun):
                            local_chunks = local_chunks - 1
                            current_encoding_length = local_chunks * chunk_size
                            encoded_length = token_index - random.randrange(0, output_model.window_length)
                            most_recent_encoded_token_position = token_index - encoded_length
                            gap_input = embedded[:, current_encoding_length:encoded_length, :]
                            gap_encoding = compressor(gap_input, device = "cuda")
                            gap_encoding = gap_encoding.unsqueeze(0) # get the dim dimension
                            previous_encoded_inputs = rearranged_encodings[:, :local_chunks, :]
                            encoded_inputs = torch.concatenate((previous_encoded_inputs, gap_encoding), axis = 1)
                        else:
                            most_recent_encoded_token_position = token_index - (local_chunks * chunk_size)
                            encoded_inputs = rearranged_encodings[:, :local_chunks, :]


                        
                    else:
                        most_recent_encoded_token_position = None
                        encoded_inputs = None
                    starting_index = max(token_index - output_model.window_length, 0) # up to window length
                    sliding_window = embedded[:, starting_index:token_index, :]
                    predictions = output_model(encoded_inputs, sliding_window, most_recent_encoded_token_position, device = "cuda")    
                    model_outputs.append(predictions)
                    
                    ground_truth.append(sequence[:, token_index]) # we're using direct indexing so we don't need to add 1

            torch.cuda.synchronize()
            
            preds = torch.cat(model_outputs, dim = 0)
            labels = torch.cat(ground_truth, dim = 0)

            loss = torch.nn.functional.cross_entropy(preds, labels)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            start = start + batch_size

    return losses



torch.cuda.empty_cache()
compressor = CompressionEncoder(encoding_dimensionality = 4096, word_embedding_dim = 1024,
                                dim = 1024, depth = 6, heads = 8, mlp_dim = 2048).cuda()
decoder = OutputModel(num_tokens = 50258, encoding_dimensionality= 4096, word_embedding_dim= 1024, 
                      dim = 4096, depth = 12, heads = 8, mlp_dim=4096, window_length = 64).cuda()

print(count_params(compressor)  + count_params(decoder))

directory = '../raw_text'
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = TextReader(directory, tokenizer=tokenizer)
dataloader = DataLoader(dataset, 1, num_workers = 1)

optimizer = torch.optim.Adam(list(compressor.parameters()) + list(decoder.parameters()))

batch_size = 64
streams = []
running_losses = []
for i in range(batch_size):
    streams.append(torch.cuda.Stream())
num_epochs = 5

for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        try:
            batch = batch.cuda()
            start = time.perf_counter()
            losses = train_step(compressor, decoder, sequence = batch, batch_size = batch_size, optimizer = optimizer, streams = streams)
            #print(time.perf_counter() - start)
            running_losses = running_losses + losses
            if i % 10 == 0:
                print(f"Loss is {listmean(running_losses)}")
            if i != 0 and i % 100 == 0:
                torch.save(compressor.state_dict(), "compressor")
                torch.save(decoder.state_dict(), "decoder")

        except Exception as e:
            print(f"Overran memory with {batch.shape}")
            print(e)
            torch.cuda.empty_cache()





'''
print(torch.cuda.mem_get_info())
print(count_params(compressor))
x = torch.zeros((16, 1024, 1024)).cuda()
start = time.perf_counter()
with torch.autocast(device_type="cuda", dtype = torch.float16):
    for i in range(10):
        compressor(x, device = "cuda")
print(time.perf_counter() - start)

'''
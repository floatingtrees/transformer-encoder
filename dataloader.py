from collections import defaultdict
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os

import os

def get_all_folders(directory):
    """
    Get a list of all folders inside the specified directory.
    """
    folders = [os.path.join(directory, folder) for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    return folders

def get_text_files_in_folders(folders):
    """
    Get all paths of text files inside each folder.
    """
    text_files = []
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.txt'):
                    text_files.append(os.path.join(root, file))
    return text_files

# Example usage
directory = '../raw_text'
tokenizer = AutoTokenizer.from_pretrained("gpt2")


class TextReader(Dataset):
    def __init__(self, path, tokenizer) -> None:
        super().__init__()
        self.folders = get_all_folders(path)
        self.files = get_text_files_in_folders(self.folders)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        with open(self.files[index], "r") as file:
            data = file.read()
        return torch.squeeze(self.tokenizer(data, return_tensors="pt").input_ids)
'''
dataset = TextReader(directory, tokenizer=tokenizer)
dataloader = DataLoader(dataset, 1, num_workers = 1)

for thing in dataloader:
    print(thing.shape)
    exit()
    '''
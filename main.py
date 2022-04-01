import json

from transformers import *
import torch
from torch import nn

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

model.eval()

def split_entity(e):
    split = []
    sp = ''
    for i in range(len(e)):
        if e[i].islower() and i + 1 < len(e) and e[i + 1].isupper():
            sp += e[i]
            split += [sp.lower()]
            sp = ''
            continue

        if e[i] == '_' or e[i] == '-':
            split += [sp.lower()]
            sp = ''
            continue
        sp += e[i]
    split += [sp.lower()]
    return split


while True:

    cm = json.loads(input())

    if cm['cmd'] == 'end':
        break
    elif cm['cmd'] == 'sim':

        with torch.no_grad():

            c1 = tokenizer(split_entity(cm['s1']), return_tensors='pt', is_split_into_words=True)
            c2 = tokenizer(split_entity(cm['s2']), return_tensors='pt', is_split_into_words=True)

            n1 = model(**c1).pooler_output
            n2 = model(**c2).pooler_output

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            output = cos(n1, n2).item()
            print(output)
import json

f = open('../models/gpt2-small/gpt2-vocab.json', 'r')
vocab = json.load(f)
f.close()
vocab["<|bos|>"] = len(vocab)
vocab["<|pad|>"] = len(vocab)
print(len(vocab))
f = open('../models/gpt2-small/vocab.json', 'w')
vocab = json.dump(vocab, f)
f.close()



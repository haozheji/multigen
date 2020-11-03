import torch
import os
import json
import logging
import csv
import itertools
from torch.utils.data import Dataset
import random

from transformers import BertTokenizer

logger = logging.getLogger()

def normalize_case(text):
    if len(text) > 1:
        try:
            normalized = text[0].upper() + text[1:].lower()
            if normalized[-1] != '.':
                normalized = normalized + '.'
        except:
            raise RuntimeError("Cannot normalize text {}".format(text))
        return normalized
    return text
    


class MHDataset(Dataset):
    def __init__(self, args, tokenizer, data_path, src_max_length=256, tgt_max_length=64, do_generate=False, max_memory_size=400, max_triple_size=800):
        self.do_generate = do_generate 
        self.args = args
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.tokenizer = tokenizer
        self.max_memory_size = max_memory_size
        self.max_triple_size = max_triple_size
        self.bos = self.tokenizer.encoder["<|bos|>"]
        self.pad = self.tokenizer.encoder["<|pad|>"]
        self.eos = self.tokenizer.encoder["<|endoftext|>"]
        self.data_path = data_path
    
    def load(self):
        data_path = self.data_path
        self.source = []
        self.source_kg = []
        self.target = []
        self.source_path = os.path.join(data_path, "source.csv")
        count = 0
        with open(self.source_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row_count, row in enumerate(csv_reader):
                self.source.append(row[1:])
    
        count = 0
        self.target_path = os.path.join(data_path, "target.csv")
        with open(self.target_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if self.do_generate:
                    self.target.append(row[1:])
                else:
                    self.target.append(row[1])
        
        count = 0
        self.concepts = []
        self.concepts_labels = []
        self.distances = [] #v3
        self.head_ids = []
        self.tail_ids = []
        self.relations = []
        self.triple_labels = []
        with open(os.path.join(data_path, self.args.graph_path), 'r') as f:
            for line in f.readlines():
                line = json.loads(line)
                assert(len(line['concepts']) == len(line['labels'])), (len(line['concepts']), len(line['labels']))
                self.concepts.append(line['concepts'])
                self.concepts_labels.append(line['labels'])
                self.distances.append(line['distances'])
                self.head_ids.append(line['head_ids'])
                self.tail_ids.append(line['tail_ids'])
                self.relations.append(line['relations'])
                self.triple_labels.append(line['triple_labels'])
    
    def __len__(self):
        return len(self.source)

    def print_features(self):
        logger.info("-"*50 + "Features" + "-"*50)
        exs = [self.__getitem__(i) for i in range(0,min(3, len(self.concepts)))]
        for ex in exs:
            logger.info("Input: {}".format([self.tokenizer.decoder[x] for x in ex[0].tolist()]))
            logger.info("Attention mask: {}".format(ex[1].tolist()))
            logger.info("Position: {}".format(ex[2].tolist()))
            logger.info("Target: {}".format([self.tokenizer.decoder[x] for x in ex[3].tolist()]))
            logger.info("Position: {}".format(ex[4].tolist()))
            logger.info("Labels: {}".format([self.tokenizer.decoder[x] for x in (ex[5].masked_select(ex[5]>=0).tolist())]))
            logger.info("Gate labels: {}".format(ex[-1].tolist()))

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]
        concept = self.concepts[idx]
        cpt_label = self.concepts_labels[idx]
        dist = self.distances[idx] #v3
        relations = self.relations[idx]
        head_ids = self.head_ids[idx]
        tail_ids = self.tail_ids[idx]
        triple_labels = self.triple_labels[idx]
        relations = [x[0] for x in relations]

        assert(len(dist) == len(concept))

        concept_ids = []
        _concept_ids = []
        concept_mask = []
        bert_input = []
        bert_mask = []
        _concept_label = cpt_label.copy()
        head_ids_trunc = head_ids.copy()
        tail_ids_trunc = tail_ids.copy()
        relations_trunc = relations.copy()
        triple_labels_trunc = triple_labels.copy()
        _distance = dist.copy()
        vocab_map = [] # usage: cpt_prob.gather(-1, vocab_map) vocab_map size the same as gpt-2 vocab
        map_mask = [] # usage: cpt_prob_vocab.masked_fill_(map_mask == 0, 0)
        target_concept_ids = []

        distance = []
        concept_label = []
        count = 0
        for e, l, d in zip(concept, _concept_label, _distance):
            tok = self.tokenizer.encode(' ' + e)
            count += 1
            if len(tok) == 1:
                _concept_ids.append(tok[0])
                concept_ids.append(tok[0])
                distance.append(d)
                concept_label.append(l)
                if l == 1:
                    #print(count)
                    target_concept_ids.append(tok[0])
        
        if len(concept_ids) > self.max_memory_size:
            concept_ids = concept_ids[:self.max_memory_size]
            concept_label = concept_label[:self.max_memory_size]
            distance = distance[:self.max_memory_size]

        while len(concept_ids) < self.max_memory_size:
            concept_ids.append(self.pad)
            concept_label.append(-1)
            distance.append(0)

        for idx in self.tokenizer.decoder.keys():
            try: 
                pos = _concept_ids.index(idx)
                vocab_map.append(pos)
                map_mask.append(1)
            except:
                vocab_map.append(0)
                map_mask.append(0)

        assert(len(vocab_map) == len(self.tokenizer.decoder)), len(vocab_map)
        assert(len(map_mask) == len(self.tokenizer.decoder)), len(map_mask)

        if len(head_ids_trunc) > self.max_triple_size:
            head_ids_trunc = head_ids_trunc[:self.max_triple_size]
            tail_ids_trunc = tail_ids_trunc[:self.max_triple_size]
            relations_trunc = relations_trunc[:self.max_triple_size]
            triple_labels_trunc = triple_labels_trunc[:self.max_triple_size]
            
        
        while len(head_ids_trunc) < self.max_triple_size:
            head_ids_trunc.append(0) 
            tail_ids_trunc.append(0) 
            relations_trunc.append(0)
            triple_labels_trunc.append(-1)
        

        src_input_ids = []
        for s in src:
            src_input_ids.extend(self.tokenizer.encode(' ' + s))
            src_input_ids.append(self.eos)
        src_position_ids = list(range(0, len(src_input_ids)))

        assert (len(src_input_ids) == len(src_position_ids))
        if len(src_input_ids) > self.src_max_length:
            src_input_ids = src_input_ids[:self.src_max_length]
            src_position_ids = src_position_ids[:self.src_max_length]

        attention_mask = [1] * len(src_input_ids)

        while len(src_input_ids) < self.src_max_length:
            src_input_ids += [self.pad]
            src_position_ids += [0]
            attention_mask += [0]
        
        target_input_ids = []
        target_position_ids = []
        labels = []
        gate_labels = []

        if not self.do_generate:
            target_input_ids = [self.bos] + self.tokenizer.encode(' ' + tgt)
            target_position_ids = list(range(0, len(target_input_ids)))
            if len(target_input_ids) > self.tgt_max_length:
                target_input_ids = target_input_ids[:self.tgt_max_length]
                target_position_ids = target_position_ids[:self.tgt_max_length]

            
            labels = target_input_ids[1:] + [self.eos]
            gate_labels = [1 if x in target_concept_ids else 0 for x in labels]

            while len(target_input_ids) < self.tgt_max_length:
                target_input_ids += [self.pad]
                target_position_ids += [0] 
                labels += [-1]
                gate_labels += [-1]
        
        gate_labels = [-1] * self.src_max_length + gate_labels
        labels = [-1] * self.src_max_length + labels

        assert(len(concept_ids) == self.max_memory_size), len(concept_ids)
        assert(len(distance) == self.max_memory_size), len(distance)
        return (torch.tensor(src_input_ids), 
                torch.tensor(attention_mask),
                torch.tensor(src_position_ids),
                torch.tensor(target_input_ids),
                torch.tensor(target_position_ids),
                torch.tensor(labels), 
                torch.tensor(concept_ids), 
                torch.tensor(concept_label),
                torch.tensor(distance),
                torch.tensor(head_ids_trunc),
                torch.tensor(tail_ids_trunc),
                torch.tensor(relations_trunc),
                torch.tensor(triple_labels_trunc),
                torch.tensor(vocab_map),
                torch.tensor(map_mask), 
                torch.tensor(gate_labels))

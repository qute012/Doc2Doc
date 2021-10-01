import random
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer


class RelDocDataset(Dataset):
    def __init__(
            self,
            path='./datasets/msmarco_data/top1000.dev',
            tokenizer: BertTokenizer = None,
            src_max_len=512,
            gen_max_len=150,
            episode=100000,
    ):
        super(RelDocDataset, self).__init__()
        self.tokenizer = tokenizer
        self.src_max_len = src_max_len
        self.gen_max_len = gen_max_len
        self.rel_map = self._prep_data(path)
        self.episode = episode

    def _prep_data(self, path):
        rel_map = {}
        with open(path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            qid, did, passage = line.split('\t', maxsplit=2)
            if not qid in rel_map.keys():
                rel_map[qid] = list()
            rel_map[qid].append(passage)
        
        qids = rel_map.keys()
        trunc_qids = []
        for qid in qids:
            if len(rel_map[qid])<2:
                trunc_qids.append(qid)
        for qid in trunc_qids:
            del rel_map[qid]
        return rel_map

    def __getitem__(self, idx):
        qid = random.choice(list(self.rel_map.keys()))
        rel_docs = self.rel_map[qid]
        src_idx, tgt_idx = random.sample(range(len(rel_docs)), 2)
        src_doc, tgt_doc = rel_docs[src_idx], rel_docs[tgt_idx]

        src = self.tokenizer(
            src_doc,
            return_tensors='pt',
            max_length=self.src_max_len,
            truncation=True
        )

        tgt = self.tokenizer(
            tgt_doc,
            return_tensors='pt',
            max_length=self.gen_max_len,
            truncation=True
        )

        src = self._src_truncate(src)
        tgt = self._gen_truncate(tgt)
        return src, tgt

    def _src_truncate(self, src):
        return {key: src.get(key)[0, :self.src_max_len] for key in src}

    def _gen_truncate(self, src):
        return {key: src.get(key)[0, :self.gen_max_len] for key in src}

    def __len__(self):
        return self.episode


def _collate_fn(batch):
    seq1 = []
    seq1_mask = []
    seq2 = []
    for s1, s2 in batch:
        seq1.append(s1.get('input_ids'))
        seq1_mask.append(s1.get('attention_mask'))
        seq2.append(s2.get('input_ids'))

    seq1 = pad_sequence(seq1).transpose(1, 0)
    seq2 = pad_sequence(seq2).transpose(1, 0)
    seq1_mask = pad_sequence(seq1_mask).transpose(1, 0)
    return {'input_ids': seq1.long(), 'attention_mask': seq1_mask.long()}, seq2.long()


def get_dataloader(path, tokenizer, src_max_len=400, gen_max_len=150, episode=100000, batch_size=4, shuffle=False, num_workers=2, pin_memory=True):
    sampler = RandomSampler(list(range(episode)))
    dataset = RelDocDataset(path, tokenizer, src_max_len, gen_max_len, episode)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, pin_memory=pin_memory, sampler=sampler,
                      num_workers=num_workers, collate_fn=_collate_fn)

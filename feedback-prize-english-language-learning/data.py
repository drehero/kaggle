import math
import re
from collections import Counter

import spacy

import readability
from spellchecker import SpellChecker

import torch
from torch.utils.data import Dataset

from knowledge import pos_map, entity_map, dependency_map, dale_chall_map


COLLATE_INPUTS = [
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "overflowing_tokens",
    "num_truncated_tokens",
    "special_tokens_mask",
    "token_level_feats"
]

def collate_fn(batch):
    mask_len = int(batch["attention_mask"].sum(axis=1).max())
    for k, v in batch.items():
        if k in COLLATE_INPUTS:
            batch[k] = v[:, :mask_len]
    return batch


def get_essay_level_features(text, nlp, spell):
    essay_level_features = []
    # punctuation
    c = Counter(text)
    essay_level_features += [c.get(",", 0), c.get("!", 0), c.get("?", 0)]

    # readability
    results = readability.getmeasures(text)
    for v in results.values():
        essay_level_features += list(v.values())

    # spelling
    essay_level_features += [len(spell.unknown(re.findall(r'\w+', text.lower())))]

    # pos
    doc = nlp(text)
    c = Counter([t.pos_ for t in doc])
    essay_level_features += [
        c.get("NOUN", 0) + c.get("PROPN", 0),  # nouns
        c.get("VERB", 0),  # verbs
        c.get("ADV", 0),  # adverbs
        c.get("ADJ", 0),  # adjectives
    ]
    
    # stopwords
    essay_level_features += [sum([t.is_stop for t in doc])]

    return torch.tensor(essay_level_features, dtype=torch.float)

class AlignedTokenizer:
    def __init__(self, tokenizer, spacy_model="en_core_web_sm"):
        self.tokenizer = tokenizer
        self.nlp = spacy.load(spacy_model)

    def encode(
        self,
        text,
        return_tensors,
        add_special_tokens,
        max_length,
        #pad_to_max_length=True,
        padding,
        truncation,
    ):
        text = " ".join(text.strip().split())
        doc = self.nlp(text)
        inputs = self.tokenizer.encode_plus(
            doc.text,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            #pad_to_max_length=True,
            padding=padding,
            truncation=truncation,
            return_offsets_mapping=True
        )
        hf_offset = inputs.pop("offset_mapping")
        spacy_offset = [(t.idx, t.idx+len(t)) for t in doc]
        s_idx = 0
        alignment = []
        for hf_start, hf_end in hf_offset:
            if hf_start == 0 and hf_end == 0:
                # special token
                alignment += [None]
            else:
                s_start, s_end = spacy_offset[s_idx]
                if hf_end == s_end:
                    alignment += [s_idx]
                    s_idx += 1
                elif hf_end < s_end:
                    alignment += [s_idx]
                else:
                    # This occures if spacy splits a word into shorter tokens
                    # than huggingface. For example: "wont" is split by spacy into
                    # ["wo", "nt"] and by hf into ["wont"]. In this case the 
                    # word piece token is mapped to the last spacy word piece.
                    # i.e. in the example above "wont" is mapped to "nt")
                    # another example is "shouldn't" which is split by hf into
                    # ["shouldn", "'", "t"] but by spacy into ["should", "n't"].
                    # Here all 3 hf token are alligned with "n't".
                    # This is not be ideal and could be improved.
                    while spacy_offset[s_idx][1] < hf_end:
                        s_idx += 1
                    alignment += [s_idx]
        spacy_tokens = [doc[i] if i is not None else None for i in alignment] 
        inputs["spacy_tokens"] = spacy_tokens
        return inputs

    def encode_plus(
        self,
        text,
        return_tensors,
        add_special_tokens,
        max_length,
        #pad_to_max_length=True,
        padding,
        truncation,
    ):
        return self.encode(
            text,
            return_tensors,
            add_special_tokens,
            max_length,
            #pad_to_max_length=True,
            padding,
            truncation,
        )


class FBData(Dataset):
    def __init__(self, df, args, essay_feats_scaler):
        super().__init__()
        self.args = args
        self.text = df["full_text"].values
        if args.train:
            self.targets = df[args.target_names].values
        if args.essay_level_feats:
            self.essay_level_feats = torch.stack(df["essay_level_feats"].to_list())
            if essay_feats_scaler is not None:
                self.essay_level_feats = torch.tensor(essay_feats_scaler.transform(self.essay_level_feats), dtype=torch.float)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if self.args.train:
            inputs = self.args.tokenizer.encode_plus(
                self.text[idx],
                return_tensors=None,
                add_special_tokens=True,
                max_length=self.args.max_len,
                #pad_to_max_length=True,
                padding="max_length",
                truncation=True
            )
        else:
            inputs = self.args.tokenizer.encode_plus(
                self.text[idx],
                return_tensors=None,
                add_special_tokens=True,
                max_length=self.args.max_len,
                padding="max_length",
                truncation=True
            )

        # TODO having this a few lines below might have caused a bug
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                inputs[k] = torch.tensor(v, dtype=torch.long)

        if self.args.token_level_feats:
            spacy_tokens = inputs.pop("spacy_tokens")
            inputs["token_level_feats"] = self.get_token_level_features(spacy_tokens)

        if self.args.essay_level_feats:
            inputs["essay_level_feats"] = self.essay_level_feats[idx]
        
        if self.args.train:
            targets = self.targets[idx]
            if self.args.scale_targets:
                targets = self.args.target_scaler.transform(targets.reshape(1, -1)).squeeze()
            targets = torch.tensor(targets, dtype=torch.float)
            return inputs, targets

        return inputs


    def get_token_level_features(self, spacy_tokens):
        n_pos_tags = len(pos_map)
        n_entities = len(entity_map)
        n_dependencies = len(dependency_map)
        n_dale_chall = 2
        pos_tensors = []
        ent_tensors = []
        dep_tensors = []
        basic_tensors = []
        for t in spacy_tokens:
            pos_tensor = torch.zeros(n_pos_tags)
            ent_tensor = torch.zeros(n_entities)
            dep_tensor = torch.zeros(n_dependencies)
            basic_tensor = torch.zeros(n_dale_chall)
            if t is not None:
                if t.pos_ in pos_map:
                    pos_tensor[pos_map[t.pos_]] = 1
                if t.ent_type_ in entity_map:
                    ent_tensor[entity_map[t.ent_type_]] = 1
                if t.dep_ in dependency_map:
                    dep_tensor[dependency_map[t.dep_]] = 1
                if t.text in dale_chall_map:
                    basic_tensor[0] = 1
                else:
                    basic_tensor[1] = 1
            pos_tensors += [pos_tensor]
            ent_tensors += [ent_tensor]
            dep_tensors += [dep_tensor]
            basic_tensors += [basic_tensor]
        pos_tensors = torch.stack(pos_tensors, dim=1)
        ent_tensors = torch.stack(ent_tensors, dim=1)
        dep_tensors = torch.stack(dep_tensors, dim=1)
        basic_tensors = torch.stack(basic_tensors, dim=1)
        return torch.cat((pos_tensors, ent_tensors, dep_tensors, basic_tensors), dim=0).transpose(0, 1)


class MultiScaleData(Dataset):
    def __init__(self, df, args):
        super().__init__()
        self.args = args
        self.text = df["full_text"].values
        if args.train:
            self.targets = df[args.target_names].values

    def __len__(self):
        return self.text.shape[0]

    def __getitem__(self, idx):
        inputs = {}
        inputs["word_doc"] = self.encode_document(self.text[idx], self.args.max_len).squeeze()
        seg_chunks = []
        for i in range(len(self.args.chunk_sizes)):
            seg_chunks += [self.encode_document(self.text[idx], self.args.chunk_sizes[i])]
        inputs["seg_chunks"] = seg_chunks

        if self.args.train:
            targets = self.targets[idx]
            if self.args.scale_targets:
                targets = self.args.target_scaler.transform(targets.reshape(1, -1)).squeeze()
            targets = torch.tensor(targets, dtype=torch.float)
            return inputs, targets
        return inputs
            

    def encode_document(self, doc, max_input_length):
        # https://github.com/lingochamp/Multi-Scale-BERT-AES/blob/main/encoder.py
        tokenized_document = self.args.tokenizer.tokenize(doc)
        max_sequences_per_document = math.ceil(self.args.max_len/(max_input_length-2))
        output = torch.zeros(size=(max_sequences_per_document, 3, max_input_length), dtype=torch.long)
        for seq_index, i in enumerate(range(0, len(tokenized_document), (max_input_length-2))):
            raw_tokens = tokenized_document[i:i+(max_input_length-2)]
            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = self.args.tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)

            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)
            output[seq_index] = torch.cat(
                (
                    torch.LongTensor(input_ids).unsqueeze(0),
                    torch.LongTensor(input_type_ids).unsqueeze(0),
                    torch.LongTensor(attention_masks).unsqueeze(0)
                ), dim=0)
        return output

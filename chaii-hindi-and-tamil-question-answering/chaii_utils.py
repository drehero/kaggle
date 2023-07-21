from chaii_config import *

import collections

import numpy as np
import pandas as pd
from nltk.metrics import jaccard_distance
from tqdm.auto import tqdm


def prepare_train_features(examples_df, tokenizer, config):
    examples = examples_df.copy().reset_index(drop=True)
    
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["question" if config["pad_on_right"] else "context"].to_list(),
        examples["context" if config["pad_on_right"] else "question"].to_list(),
        truncation="only_second" if config["pad_on_right"] else "only_first",
        max_length=config["max_length"],
        stride=config["doc_stride"],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=True,
        padding="max_length",
    )

    # Example to feature mapping as long contexts might give multiple features
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # Character to token mapping
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Get labels
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Sequence ids indicate from which sequence a token is
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]  # which example created this feature
        answer_text = examples["answer_text"].values[sample_index]
        answer_start_char = examples["answer_start"].values[sample_index]
        answer_end_char = answer_start_char + len(answer_text)

        # Find start and end token index (set default to cls index)
        answer_start_token = cls_index
        answer_end_token = cls_index

        # Get start and end of context
        token_start_index = 0
        while sequence_ids[token_start_index] != (1 if config["pad_on_right"] else 0):
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != (1 if config["pad_on_right"] else 0):
            token_end_index -= 1

        # Detect if the answer is inside the span (otherwise use leave cls label)
        if (offsets[token_start_index][0] <= answer_start_char and offsets[token_end_index][1] >= answer_end_char):
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start_char:
                token_start_index += 1
            answer_start_token = token_start_index - 1
            while offsets[token_end_index][1] >= answer_end_char:
                token_end_index -= 1
            answer_end_token = token_end_index + 1

        if answer_start_token == cls_index or answer_end_token == cls_index:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            tokenized_examples["start_positions"].append(answer_start_token)
            tokenized_examples["end_positions"].append(answer_end_token)

    return tokenized_examples


def prepare_validation_features(examples_df, tokenizer, config):
    examples = examples_df.copy().reset_index(drop=True)
    
    examples["question"] = [q.lstrip() for q in examples["question"]]
    tokenized_examples = tokenizer(
        examples["question" if config["pad_on_right"] else "context"].to_list(),
        examples["context" if config["pad_on_right"] else "question"].to_list(),
        truncation="only_second" if config["pad_on_right"] else "only_first",
        max_length=config["max_length"],
        stride=config["doc_stride"],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=True,
        padding="max_length",
    )
    
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []
    
    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if config["pad_on_right"] else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    return tokenized_examples

    
def postprocess_predictions(examples_df, tokenized_features, raw_predictions, tokenizer, verbose=True):
    examples = examples_df.copy().reset_index(drop=True)
    all_start_probs, all_end_probs = raw_predictions
    
    # Map examples to its corresponding features
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, example_id in enumerate(tokenized_features["example_id"]):
        features_per_example[example_id_to_index[example_id]].append(i)    
    predictions = collections.OrderedDict()
    
    if verbose:
        print(f"Post-processing {examples.shape[0]} example predictions split into {len(tokenized_features['example_id'])} features.")
    for example_index, example in (tqdm(examples.iterrows(), total=examples.shape[0]) if verbose else examples.iterrows()):
        feature_indices = features_per_example[example_index]
        
        #min_null_score = None
        valid_answers = []
        
        context = example["context"]
        # Loop over all features associated with the example
        for feature_index in feature_indices:
            start_probs = all_start_probs[feature_index]
            end_probs = all_end_probs[feature_index]
            
            offset_mapping = tokenized_features["offset_mapping"][feature_index]
            
            # Update minimum null prediction
            #cls_index = tokenized_features["input_ids"][feature_index].index(tokenizer.cls_token_id)
            #feature_null_score = start_probs[cls_index] + end_probs[cls_index]
            #if min_null_score is None or min_null_score < feature_null_score:
            #    min_null_score = feature_null_score
            
            # Go through all possibilities for the N_BEST_SIZE greater start and end probs
            start_indices = np.argsort(start_probs)[-1: -N_BEST_SIZE-1: -1].tolist()
            end_indices = np.argsort(end_probs)[-1: -N_BEST_SIZE-1: -1].tolist()
            for start_index in start_indices:
                for end_index in end_indices:
                    # Dont consider out of scope indices, either because the indices are out of bounds or
                    # correspond to part of the input_ids that are not in the context
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Dont consider answers with length < 0 or > MAX_ANSWER_LENGTH
                    if end_index < start_index or end_index - start_index + 1 > MAX_ANSWER_LENGTH:
                        continue
                    
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_probs[start_index] + end_probs[end_index],
                            "text": context[start_char: end_char],
                        }
                    )
            
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # Create default prediction
            best_answer = {"score": 0.0, "text": ""}
        
        #if not SQUAD_V2:
        predictions[example["id"]] = best_answer["text"]
        #else:
        #    anwer = best_answer["text"] if best_answer["score"] > min_null_score else ""
        #    predictions[example["id"]] = answer
            
    return predictions


def create_model_input(tokenized_features, is_train=True):
    if is_train:
        X_train = [
            np.array(tokenized_features["input_ids"]),
            np.array(tokenized_features["attention_mask"]),
            np.array(tokenized_features["token_type_ids"]),
        ]
        Y_train = [
            np.array(tokenized_features["start_positions"]),
            np.array(tokenized_features["end_positions"]),
        ]
        return X_train, Y_train
    else:
        X_test = [
            np.array(tokenized_features["input_ids"]),
            np.array(tokenized_features["attention_mask"]),
            np.array(tokenized_features["token_type_ids"]),
        ]
        return X_test


def jaccard_similarity(str1, str2):
    str1 = set(str1.split())
    str2 = set(str2.split())
    return 1 - jaccard_distance(str1, str2)


def qa_json_to_df(json_data, language=np.nan):
    examples = []
    data = json_data["data"]
    for article in data:
        paragraphs = article["paragraphs"]
        for context_qas in paragraphs:
            context = context_qas["context"]
            qas = context_qas["qas"]
            for qa in qas:
                question = qa["question"]
                answer_text = qa["answers"][0]["text"]
                answer_start = qa["answers"][0]["answer_start"]
                id_ = qa["id"]
                examples.append(
                    {
                        "id": id_,
                        "context": context,
                        "question": question,
                        "answer_text": answer_text,
                        "answer_start": answer_start,
                        "language": language,
                    }
                )
    return pd.DataFrame(examples)

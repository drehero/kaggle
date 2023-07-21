import os
os.system("pip install -U --no-build-isolation --no-deps ../input/transformers-master/ -qq")

import transformers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from chaii_config import *


XLM_ROBERTA_CONFIG = {
    # model
    "model_checkpoint": "deepset/xlm-roberta-large-squad2",
    "model_name": "xlm-roberta-large-squad2",
    "model_type": "xlm_roberta",
    
    # tokenizer
    "tokenizer_name": "deepset/xlm-roberta-large-squad2",
    "max_length": 384,
    "doc_stride": 128,
    "pad_on_right": True,
    
    # train
    "batch_size": 32 if USE_TPU else 4,
    "learning_rate": 3e-5,
}
MURIL_CONFIG = {
    # model
    "model_checkpoint": "google/muril-large-cased",
    "model_name": "muril-large-cased",
    "model_type": "bert",
    
    # tokenizer
    "tokenizer_name": "google/muril-large-cased",
    "max_length": 384,
    "doc_stride": 128,
    "pad_on_right": True,
    
    # train
    "batch_size": 32 if USE_TPU else 4,
    "learning_rate": 3e-5,
}
REMBERT_CONFIG = {
    # model
    "model_checkpoint": "google/rembert",
    "model_name": "rembert",
    "model_type": "rembert",
    
    # tokenizer
    "tokenizer_name": "google/rembert",
    "max_length": 384,
    "doc_stride": 128,
    "pad_on_right": True,
    
    # train
    "batch_size": 32 if USE_TPU else 4,
    "learning_rate": 3e-5,
}
MODEL_CONFIGS = [XLM_ROBERTA_CONFIG, MURIL_CONFIG, REMBERT_CONFIG]

        

def create_model(config):
    input_ids = layers.Input(shape=(config["max_length"],), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(config["max_length"],), dtype=tf.int32)
    attention_mask = layers.Input(shape=(config["max_length"],), dtype=tf.int32)
    
    if config["model_type"] == "xlm_roberta":
        encoder = transformers.TFRobertaForQuestionAnswering.from_pretrained(config["model_checkpoint"], from_pt=True)
        embedding = encoder.roberta(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )[0]
    elif config["model_type"] == "bert":
        encoder = transformers.TFBertForQuestionAnswering.from_pretrained(config["model_checkpoint"], from_pt=True)
        embedding = encoder.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )[0]
    elif config["model_type"] == "rembert":
        encoder = transformers.TFRemBertForQuestionAnswering.from_pretrained(config["model_checkpoint"], from_pt=True)
        embedding = encoder.rembert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )[0]
    
    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = layers.Flatten()(end_logits)

    model = keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=[start_logits, end_logits]
    )

    optimizer = keras.optimizers.Adam(learning_rate=config["learning_rate"])
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model



if __name__ == "__main__":
    # Save tokenizers and models
    for model_config in MODEL_CONFIGS:
        if model_config["model_type"] == "xlm_roberta":
            tokenizer = transformers.XLMRobertaTokenizerFast.from_pretrained(model_config["tokenizer_name"])
        elif model_config["model_type"] == "bert":
            tokenizer = transformers.BertTokenizerFast.from_pretrained(model_config["tokenizer_name"])
        elif model_config["model_type"] == "rembert":
            tokenizer = transformers.RemBertTokenizerFast.from_pretrained(model_config["tokenizer_name"])
            
        tokenizer_path = f"{model_config['model_name']}-tokenizer"
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Written tokenizer of model {model_config['model_name']} to {tokenizer_path}.")
        
        model = create_model(model_config)
        model_path = f"{model_config['model_name']}-model"
        #save_locally = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
        model.save(model_path, save_format="h5")#, options=save_locally)
        print(f"Written model {model_config['model_name']} to {model_path}.")


# %% [code]

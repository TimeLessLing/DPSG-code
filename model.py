import os
from transformers import AutoTokenizer

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import T5Tokenizer
# from t5_modeling_lexical import T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoModelForSeq2SeqLM

# Use tokenizer.add_special_tokens, and model.resize_token_embeddings to add special tokens
# This will return the tokenizer and the model
# If we get a brand new model (without fine-tuning), we add new_tokens to the tokenizer and the embedding layer of the model
# Else, we load the model from the given file_dir
def get_model(model_type, model_name, plm_path, tokenizer_name, new_tokens:list = []):
    if model_type in ['t5', 'mt5']:
        model_cls = T5ForConditionalGeneration
        tokenizer_cls = T5Tokenizer
    elif model_type in ['bart']:
        model_cls = BartForConditionalGeneration
        tokenizer_cls = BartTokenizer
    else:
        raise ValueError("Unsupported Model")

    # model = AutoModelForSeq2SeqLM.from_pretrained(plm_path)
    model = model_cls.from_pretrained(plm_path)
    # original_config = model.config
    # model = Model(original_config, model_type, plm_path, 30, 768)
    # model = Model(model_type, plm_path, 30, 768)
    # model = model.cuda()
    tokenizer = tokenizer_cls.from_pretrained(plm_path)
    for token in new_tokens:
        tokenizer.add_tokens(token, special_tokens = True)
    if len(new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))
        # model.model.resize_token_embeddings(len(tokenizer))

    config = model.config
    # config = model.model.config
    return model, tokenizer, config


def save_model(model, tokenizer, save_dir = "ckpt"):
    model_path = os.path.join(save_dir, "model")
    tokenizer_path = os.path.join(save_dir, "tokenizer")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)



if __name__ == "__main__":
    model, tokenizer, config = get_model("t5-base", "t5-base")
    save_model(model, tokenizer, "ckpt/test")
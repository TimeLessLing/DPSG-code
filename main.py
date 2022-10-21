import json
import os
import dataclasses
import importlib
from dataclasses import dataclass, field
from importlib_metadata import metadata
import torch
from torch.utils.data import dataset

from transformers import Seq2SeqTrainingArguments as HfTrainingArguments
from transformers import HfArgumentParser
from transformers import set_seed

import deepspeed
# Define arguments for Model, Data, and Training


from model import get_model, save_model
from dataset import DataLoader, DataLoaderX, Seq2SeqDataset
from trainer import Seq2SeqTrainer

@dataclass
class ModelArguments:
    conf_dir: str = field(
        metadata={"help": "directory of the configuration file, it should include the evaluation metric function, and the special token list (Optional)"}
    )
    model_type: str = field(
        default="t5",
        metadata={"help": "choose from t5 and bart"}
    )
    model_name: str = field(
        default="t5-base",
        metadata={"help": "Specify Which model we use, T5 or BART"}
    )
    plm_path: str = field(
        default= "/dssg/home/ai2010813738/students/lbd2020/Data/Pretrain/T5v1.1-base",
        metadata={"help": "the path to load pretrain plm"}
    )
    tokenizer_name: str = field(
        default="t5-base",
        metadata={"help": "Specify Which tokenizer we use, T5 or BART"}
    )
    model_id: str = field(
        default="31773",
        metadata={"help": "Model ID to avoid output races"}
    )
    add_special_token: bool = field(
        default=False,
        metadata={"help": "Whether Need to add special tokens importing from the configuration file"}
    )
    assign_token_embedding: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize the embedding of some tokens from another tokens"}
    )

@dataclass
class DataArguments:
    train_dir: str = field(
        metadata={"help": "training set directory"}
    )
    eval_dir: str = field(
        metadata={"help": "validation set directory"}
    )
    test_dir: str = field(
        metadata={"help": "test set directory"}
    )
    max_input_length: int = field(
        default=512, 
        metadata={"help": "max length of input sequence after tokenization"}
    )
    max_output_length: int = field(
        default=512, 
        metadata={"help": "max length of input sequence after tokenization"}
    )

@dataclass
class TrainingArguments(HfTrainingArguments):
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )
    # Turn off train/test
    seed: int = field(
        default=42,
        metadata={"help": "set seed for reproducibility"}
    )

    do_train: bool = field(
        default=True,
        metadata={"help": "training"}
    )

    do_predict: bool = field(
        default=True,
        metadata={"help": "predication"}
    )




parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, train_args = parser.parse_args_into_dataclasses()
set_seed(train_args.seed)
train_args.generation_max_length = data_args.max_output_length
train_args.disable_tqdm = True
print(data_args.max_input_length, data_args.max_input_length)


# Refine the output dir with the model ID
train_args.output_dir = train_args.output_dir + model_args.model_id

# Import Task Configurations, mainly the metric, and the special tokens
task_config = importlib.import_module(model_args.conf_dir)

# Get the model, and the tokenizer
if model_args.add_special_token:
    special_tokens = task_config.special_tokens
    print(task_config.__dir__())
    print("Add Special Tokens!")
    print(special_tokens)
else:
    special_tokens = []
model, tokenizer, model_config = get_model(model_args.model_type, model_args.model_name, model_args.plm_path, model_args.tokenizer_name, special_tokens)

if model_args.assign_token_embedding:
    origin_tokens = task_config.origin_tokens
    target_tokens = task_config.target_tokens

    for o, t in zip(origin_tokens, target_tokens):
        # o_token_ids = [tokenizer.convert_tokens_to_ids([item])[0] for item in o]
        if len(o) == 1:
            o_tokens = tokenizer.tokenize(o[0])
            o_token_ids = tokenizer.convert_tokens_to_ids(o_tokens)
            # print(o, tokenizer.convert_tokens_to_ids(o))
            if isinstance(o_token_ids, list):
                o_token_ids = [o_token_ids]
            else:
                o_token_ids = [[o_token_ids]]
        else:
            o_token_ids = []
            for item in o:
                # print("multi-", item)
                item_tokens = tokenizer.tokenize(item)
                item_token_ids = tokenizer.convert_tokens_to_ids(item_tokens)
                # print(item_token_ids)
                if isinstance(item_token_ids, list):
                    o_token_ids.append(item_token_ids)
                else:
                    o_token_ids.append([item_token_ids])
        o_counts = len(o)

        t_token_id = tokenizer.convert_tokens_to_ids([t])[0]
        # print(o)
        # print(o_token_ids)

        with torch.no_grad():
            print(f"Assign [[%s]] to [[%s]]" % (o, t))
            if model_args.model_type in ['bart']:
                if o_counts == 1:
                    encoder_token_embeds = 0
                    decoder_token_embeds = 0
                    for o_token_id in o_token_ids[0]:
                        encoder_token_embeds += model.model.encoder.embed_tokens.weight[o_token_id].detach().clone()
                        decoder_token_embeds += model.model.decoder.embed_tokens.weight[o_token_id].detach().clone()
                    model.model.encoder.embed_tokens.weight[t_token_id] = encoder_token_embeds
                    model.model.decoder.embed_tokens.weight[t_token_id] = decoder_token_embeds
                else:
                    encoder_token_embeds = 0
                    decoder_token_embeds = 0
                    for item in o_token_ids:
                        for o_token_id in item:
                            encoder_token_embeds += model.model.encoder.embed_tokens.weight[o_token_id].detach().clone()
                            decoder_token_embeds += model.model.decoder.embed_tokens.weight[o_token_id].detach().clone()
                    encoder_token_embeds = encoder_token_embeds / o_counts
                    decoder_token_embeds = decoder_token_embeds / o_counts
                    model.model.encoder.embed_tokens.weight[t_token_id] = encoder_token_embeds
                    model.model.decoder.embed_tokens.weight[t_token_id] = decoder_token_embeds
                # model.model.encoder.embed_tokens.weight[t_token_id] = model.model.encoder.embed_tokens.weight[o_token_id].detach().clone()
                # model.model.decoder.embed_tokens.weight[t_token_id] = model.model.decoder.embed_tokens.weight[o_token_id].detach().clone()
            elif model_args.model_type in ['t5', 'mt5']:
                if o_counts == 1:
                    # [RB] -> adverb -> a d verb
                    encoder_token_embeds = 0
                    decoder_token_embeds = 0
                    for o_token_id in o_token_ids[0]:
                        encoder_token_embeds += model.encoder.embed_tokens.weight[o_token_id].detach().clone()
                        decoder_token_embeds += model.decoder.embed_tokens.weight[o_token_id].detach().clone()
                    model.encoder.embed_tokens.weight[t_token_id] = encoder_token_embeds
                    model.decoder.embed_tokens.weight[t_token_id] = decoder_token_embeds
                else:
                    encoder_token_embeds = 0
                    decoder_token_embeds = 0
                    for item in o_token_ids:
                        for o_token_id in item:
                            encoder_token_embeds += model.encoder.embed_tokens.weight[o_token_id].detach().clone()
                            decoder_token_embeds += model.decoder.embed_tokens.weight[o_token_id].detach().clone()
                    encoder_token_embeds = encoder_token_embeds / o_counts
                    decoder_token_embeds = decoder_token_embeds / o_counts
                    model.encoder.embed_tokens.weight[t_token_id] = encoder_token_embeds
                    model.decoder.embed_tokens.weight[t_token_id] = decoder_token_embeds
            


train_dataset = Seq2SeqDataset(data_args.train_dir, data_args, tokenizer, "train")
eval_dataset = Seq2SeqDataset(data_args.eval_dir, data_args, tokenizer, "eval")
test_dataset = Seq2SeqDataset(data_args.test_dir, data_args, tokenizer, "test")

# Trainer
trainer = Seq2SeqTrainer(
    model = model,
    args = train_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = task_config.metric,
    data_collator = train_dataset.collate_fn,
    tokenizer = tokenizer
)


if train_args.do_train:
    # Model training
    trainer.train()
    print(trainer.state.best_model_checkpoint)


if train_args.do_predict:
    # Predict with the best model
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = train_args.per_device_eval_batch_size,
        collate_fn = test_dataset.collate_fn
    )

    gen_kwargs = {
        "max_length": train_args.generation_max_length if train_args.generation_max_length is not None else trainer.model.config.max_length,
        "num_beams": train_args.generation_num_beams if train_args.generation_num_beams is not None else trainer.model.config.num_beams,
    }

    result = list()
    trainer.model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = trainer._prepare_inputs(batch)
            batch_output = trainer.model.generate(
                **batch, 
                **gen_kwargs
            )

            predict_sequence = [tokenizer.decode(x).replace('<pad>', '').replace('</s>', '').replace('<s>', '').strip() for x in batch_output]
            result += predict_sequence

    predict_output_file = open(os.path.join(train_args.output_dir, "prediction.json") , "w")
    json.dump(result, predict_output_file)
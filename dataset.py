import json
import torch
from torch.utils import data
from transformers import AutoTokenizer

class Seq2SeqDataset(torch.utils.data.Dataset):
    #   Train, Eval  :
    ### input, output, xxx  ###
    #   Test        :
    ### input               ###
    def __init__(self, file_name, data_args, tokenizer : AutoTokenizer, mode = "train"):
        super().__init__()
        self.data = json.load(open(file_name))
        self.size = len(self.data)
        
        self.tokenizer = tokenizer
        self.mode = mode
        self.data_args = data_args

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        data_slice = self.data[i]
        output_item = dict()

        input_seq = data_slice['input']
        output_item['input'] = input_seq
        # output_item['lexical'] = data_slice['lexical']

        if self.mode != "test":
            output_seq = data_slice['output']
            output_item['output'] = output_seq

        for k in data_slice.keys():
            if k not in ['input', 'output']:
            # if k not in ['input', 'output', 'lexical']:
                output_item[k] = data_slice[k]

        return output_item

    def collate_fn(self, data_batch):
        output_batch = dict()

        input_seqs = [x['input'] for x in data_batch]
        input_after_tokenizer = self.tokenizer(
            input_seqs, 
            return_tensors="pt", 
            padding='longest',
            max_length = self.data_args.max_input_length,
            truncation = True,
        )
        # input_lexicals = [x['lexical'] for x in data_batch]
        # input_lexicals_after_tokenizer = self.tokenizer(
        #     input_lexicals,
        #     return_tensors="pt", 
        #     padding="longest",
        #     max_length=self.data_args.max_input_length,
        #     truncation=True,
        # )
        input_tokens = input_after_tokenizer.input_ids
        input_attens = input_after_tokenizer.attention_mask
        # input_lexicals = input_lexicals_after_tokenizer.input_ids
        output_batch['input_ids'] = input_tokens
        output_batch['attention_mask'] = input_attens
        # output_batch['lexical_ids'] = input_lexicals
        # output_batch['input_tokens'] = [self.tokenizer.tokenize(x) for x in input_seqs] 

        # if self.mode != "test":
        if 'output' in data_batch[0]:
            output_seqs = [x['output'] for x in data_batch]
            output_after_tokenizer = self.tokenizer(
                output_seqs, 
                return_tensors="pt", 
                padding='longest',
                max_length = self.data_args.max_output_length,
                truncation = True,
            )
            output_tokens = output_after_tokenizer.input_ids
            output_tokens[output_tokens == self.tokenizer.pad_token_id] = -100
            output_batch['labels'] = output_tokens
            # output_attens = output_after_tokenizer.attention_mask
            # output_batch['decoder_input_ids'] = output_tokens
            # output_batch['decoder_attention_mask'] = output_attens
            # output_batch['output_tokens'] = [self.tokenizer.tokenize(x) for x in output_seqs] 

        for k in data_batch[0].keys():
            if k not in ['input', 'output']:
            # if k not in ['input', 'output', 'lexical']:
                output_batch[k] = [x[k] for x in data_batch]
        # print(output_batch.keys())
        return output_batch




from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())








if __name__ == "__main__":
    from transformers import HfArgumentParser
    from dataclasses import dataclass, field
    @dataclass
    class DataArguments:
        """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
        """
        max_input_length: int = field(
            default=8, metadata={"help": "max length of input sequence after tokenization"}
        )
        max_output_length: int = field(
            default=8, metadata={"help": "max length of input sequence after tokenization"}
        )
    parser = HfArgumentParser(DataArguments)
    data_args, = parser.parse_args_into_dataclasses()





    split_token = '<fun_spt>'
    arg_token = '<fun_arg>'
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    tokenizer.add_tokens(split_token, special_tokens=True)
    tokenizer.add_tokens(arg_token, special_tokens=True)

    dataset = Seq2SeqDataset(file_name="data/kqapro_program/train.json", data_args = data_args, tokenizer = tokenizer, mode = "train")
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn, num_workers = 4)
    data_loaderx = DataLoaderX(dataset, batch_size=16, shuffle=True, collate_fn=dataset.collate_fn, num_workers = 4)

    for batch in data_loader:
        print(batch)
        exit()

    import time
    import torch

    start_time = time.time()
    for batch in data_loader:
        mat = torch.randn((200, 200))
        mat = mat.mm(mat)
    print("Without Multi Thread:", time.time() - start_time)
        
    start_time = time.time()
    for batch in data_loaderx:
        mat = torch.randn((200, 200))
        mat = mat.mm(mat)
    print("With Multi Thread:", time.time() - start_time)
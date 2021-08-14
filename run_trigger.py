from utils import *
from model import *
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm_notebook as tqdm
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_labels",default=3,type = int)
parser.add_argument("-b", "--batch_size",default=32,type = int)
parser.add_argument("-e", "--epoch", default=20 ,type = int)
parser.add_argument("-l","--lr", default=0.0001, type = float)
parser.add_argument("-m","--max_length",default = 256, type = int)
parser.add_argument("-p","--path",default = "ace_2005/data/Arabic", type = str)
args = parser.parse_args()
model_name = "asafaya/bert-base-arabic"

def init_bert(name):   
    tokenizer = BertTokenizer.from_pretrained(name)
    model = BertModel.from_pretrained(name)
    return tokenizer, model

def init_device():
    if torch.cuda.is_available():       
        device = torch.device("cuda:0")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

def trigger_data_further_process(input):
    notes = input['notes']
    input_notes = []
    for i, note in enumerate(notes):
        if i == 0:
            if "T" in notes[i]:
                input_notes.append("B-T")
            else:
                input_notes.append("O")
        else:
            if "T" in notes[i]:
                if "T" not in notes[i-1]:
                    input_notes.append("B-T")
                else:
                    input_notes.append("I-T")
            else:
                input_notes.append("O")
    return input_notes

class ace_dataset(Dataset):
    def __init__(self, datas, max_length, filter = False):
        if filter:
            datas = [x for x in datas if len(x['tokens']) <= max_length - 2]
        tokens = [x['tokens'] for x in datas]
        self.ids = [[2] + tokenizer.convert_tokens_to_ids(seq) + [3] for seq in tokens]
        labels = [trigger_data_further_process(x) for x in datas]
        self.map = {"O":0,"B-T":1,"I-T":2}
        self.labels = [[0]+[self.map[x] for x in y]+[0] for y in labels]
        self.len = len(self.ids)
        self.max_length = max_length
        self.tokens = tokens
        self.datas = datas
    def __getitem__(self,index):
        if type(index) == slice:
            id_torch = [torch.tensor(x) for x in self.ids[index]]
            label_torch  = [torch.tensor(x) for x in self.labels[index]]
            return pad_sequence(id_torch, batch_first = True)[:,:self.max_length],pad_sequence(label_torch, batch_first = True)[:,:self.max_length]
        else:
            id_torch = torch.tensor(self.ids[index])
            label_torch  = torch.tensor(self.labels[index])
            return id_torch[:self.max_length],label_torch[:self.max_length]
    def __len__(self):
        return self.len

def init_dataset(max_length,tokenizer,path):
    test_files, dev_files, train_files = get_data_paths(path)
    train_processor = ara_ace_master(train_files, tokenizer)
    dev_processor = ara_ace_master(dev_files, tokenizer)
    test_processor = ara_ace_master(test_files, tokenizer)
    train_processor.extract_trigger_data()
    train_datas = [y for x in train_processor.trigger_datas for y in x]
    dev_processor.extract_trigger_data()
    dev_datas = [y for x in dev_processor.trigger_datas for y in x]
    test_processor.extract_trigger_data()
    test_datas = [y for x in test_processor.trigger_datas for y in x]
    train_dataset =  ace_dataset(train_datas,max_length,filter = True)
    dev_dataset = ace_dataset(dev_datas,max_length,filter = True)
    test_dataset  = ace_dataset(test_datas,max_length,filter = True)
    return train_dataset, dev_dataset, test_dataset

def init_dataloader(train_dataset, dev_dataset, test_dataset, batch_size, max_length):
    #func = lambda x:(pad_sequence([torch.tensor(seq) for seq in x[0]],batch_first=True)[:,:max_length],pad_sequence([torch.tensor(seq) for seq in x[1]],batch_first=True)[:,:max_length])
    train_loader = DataLoader(train_dataset,batch_size = batch_size, shuffle = True, collate_fn = pad_collate(max_length))
    dev_loader= DataLoader(dev_dataset,batch_size = 32, shuffle = True, collate_fn = pad_collate(max_length))
    test_loader = DataLoader(test_dataset,batch_size = 32, shuffle = True, collate_fn = pad_collate(max_length))
    return train_loader, dev_loader, test_loader

def main():
    ace2005_path = args.path
    num_labels = args.num_labels
    batch_size = args.batch_size
    lr = args.lr
    max_length = args.max_length
    epoch = args.epoch
    tokenizer, model = init_bert(model_name)
    device = init_device()
    train_set, dev_set, test_set = init_dataset(max_length = max_length, tokenizer = tokenizer, path = ace2005_path)
    show_length_distribution(train_set)
    train_loader, _, _ = init_dataloader(train_set, dev_set, test_set, batch_size = batch_size, max_length = max_length)
    settings = {"model":model,"lr":lr,"device":device,"num_label":num_labels}
    trainer = main_trainer(**settings)
    trainer.train(train_loader,epoch,dev_set)
    output_model_file = "arabic_trigger_identifier/model.bin"
    output_config_file = "arabic_trigger_identifier/.bin"
    torch.save(trainer.model.state_dict(), output_model_file)
    trainer.model.config.to_json_file(output_config_file)

if __name__ == "__main__":
    main()


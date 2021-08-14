import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import BertTokenizer, BertForTokenClassification, BertModel
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch import nn

class focal_loss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2 , size_average=True):
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)
        self.gamma = gamma
    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = preds
        #preds_softmax = F.softmax(preds, dim=1) 
        preds_logsoft = torch.log(preds_softmax)
        #focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class Tr_indentify(nn.Module):
    def __init__(self,bert_model,num_label,device):
        super(Tr_indentify, self).__init__()
        self.model = bert_model
        self.model = self.model.to(device)
        self.hidden_size = self.model.config.hidden_size
        self.num_label = num_label
        self.fc_logits = nn.Linear(self.hidden_size, self.num_label, bias = True)
        self.device = device
        self.loss_func = focal_loss(self.num_label)
    def forward(self,input_ids, attention_mask, labels):
        bert_output = self.model(input_ids = input_ids, attention_mask = attention_mask)
        hidden_state = bert_output.last_hidden_state
        trigger_logits = self.fc_logits(hidden_state)
        trigger_softmax = F.softmax(trigger_logits, dim = -1)
        loss = self.loss_func(trigger_softmax, labels)
        return trigger_softmax, loss
    def inferer(self,input,tokenizer):
        self.eval()
        input_tokens = tokenizer.tokenize(input)
        input_ids = tokenizer.encode(input,return_tensors="pt")
        input_ids = input_ids.to(self.device)
        #input_ids = input_ids.to(device)
        bert_output = self.model(input_ids = input_ids, attention_mask = None)
        #bert_output = trainer.model.model(input_ids = input_ids, attention_mask = None)
        hidden_state = bert_output.last_hidden_state
        trigger_logits = self.fc_logits(hidden_state)
        #trigger_logits = trainer.model.fc_logits(hidden_state)
        trigger_hat = trigger_logits.cpu().detach().argmax(dim = -1)
        trigger_hat = trigger_hat.squeeze()
        trigger_hat = trigger_hat[1:-1]
        return input_tokens,trigger_hat

class main_trainer:
    def __init__(self,model,lr,device,num_label):
        self.model = Tr_indentify(model,num_label,device)
        self.lr = lr
        self.optim = Adam(self.model.parameters(),lr = self.lr)
        self.device = device
        self.model = self.model.to(self.device)
    def train(self,dataloader,epochs,eval_dataset):
        epoch_bar = tqdm(total = epochs, desc = "epoch")
        self.hist_train = []
        self.hist_eval = []
        for epoch in range(epochs):
            epoch_bar.set_postfix(epoch = epoch)
            train_bar = tqdm(total = len(dataloader),desc="train")
            epoch_pred = []
            epoch_true = []
            running_loss = 0
            for i, (ids, labels, masks) in enumerate(dataloader):
                self.model.train()
                self.optim.zero_grad()
                batch_size = ids.size()[0]
                ids = ids.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)
                trigger_softmax, loss = self.model(input_ids = ids, attention_mask = masks, labels = labels)
                #except:
                #    print(ids.shape,masks.shape,labels.shape)
                #    continue
                loss.backward()
                self.optim.step()
                y_pred = torch.argmax(trigger_softmax.cpu().detach(), dim = -1)[masks == 1]
                labels = labels.cpu().detach()
                y = labels[masks == 1]
                if len(set(y_pred.tolist())) == 1:
                    wrong = "error"
                else:
                    wrong = "none"
                epoch_pred += y_pred
                epoch_true += y
                epoch_acc = accuracy_score(epoch_true, epoch_pred)
                epoch_f1 = f1_score(epoch_true, epoch_pred,average='micro')
                train_bar.set_postfix(
                                    wrong = wrong,
                                    acc = epoch_acc,
                                    f1 = epoch_f1,
                                    loss = loss.item())
                running_loss += (loss.item() - running_loss)/(i+1)
                train_bar.update()
            train_bar.n = 0
            epoch_bar.update()
            self.hist_train.append((epoch_acc, epoch_f1, running_loss))
            eval_acc, eval_f1, eval_loss = self.eval(eval_dataset)
            self.hist_eval.append((eval_acc, eval_f1, eval_loss))
            epoch_bar.set_postfix(eval_acc = eval_acc, eval_f1 = eval_f1)
    def eval(self,dataset):
        self.model.eval()
        pred = []
        true = []
        running_loss = 0
        bar = tqdm(total = len(dataset), desc = "evaluating")
        for i, (ids, labels) in enumerate(dataset):
            ids = ids.to(self.device).unsqueeze(0)
            labels = labels.to(self.device).unsqueeze(0)
            trigger_softmax, loss = self.model(input_ids = ids, attention_mask= None,labels = labels)
            trigger_softmax = trigger_softmax.cpu().detach()
            y_pred = torch.argmax(trigger_softmax, dim = -1)
            y_pred = y_pred.squeeze()
            labels = labels.cpu().detach()
            labels = labels.squeeze()
            if len(set(y_pred.tolist())) == 1:
                error = "error"
            else:
                error = "none"
            pred += y_pred
            true += labels
            acc = accuracy_score(true, pred)
            f1 = f1_score(true, pred, average='micro')
            running_loss = (loss.item() - running_loss)/(i+1)
            bar.set_postfix(error = error, acc = acc, f1 = f1, loss = running_loss)
            bar.update()
        return acc, f1, running_loss
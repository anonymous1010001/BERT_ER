import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import copy
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
#from datasets import load_dataset, load_metric
import numpy as np
from sklearn.model_selection import train_test_split

#claim, each evidence pair csv file full data
#create the csv file from generation_pickle_to_csv_convert.py

dev_whole_data=pd.read_csv(path_to_csv_valid_file)
test_whole_data=pd.read_csv(path_to_csv_test_file)
train_whole_data=pd.read_csv(path_to_csv_train_file)

dev_whole_data_cl_id=list(dev_whole_data['Claim_id'])
dev_whole_data_cl=list(dev_whole_data['Claim'])
dev_whole_data_evi_id=list(dev_whole_data['Evi_id'])
dev_whole_data_evi=list(dev_whole_data['Evidence'])
dev_whole_data_prob=list(dev_whole_data['Prob_sc'])
dev_whole_label=list(dev_whole_data['label'])

class CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model='bert-base-cased'):
        super(CustomDataset, self).__init__()
        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

      # try:
      sent1 = str(self.data.loc[index, 'Claim'])
      sent2 = str(self.data.loc[index, 'Evidence'])

      # Tokenize the pair of sentences to get token ids, attention masks and token type ids
      encoded_pair = self.tokenizer(sent1, sent2,
                                    padding='max_length',  # Pad to max_length
                                    truncation=True,  # Truncate to max_length
                                    max_length=self.maxlen,
                                    return_tensors='pt')  # Return torch.Tensor objects

      token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
      attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
      token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

      if self.with_labels:  # True if the dataset has labels
          label = self.data.loc[index, 'label']
          return token_ids, attn_masks, token_type_ids, label
      else:
          return token_ids, attn_masks, token_type_ids


class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model="bert-base-cased", freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)

        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        if bert_model == "albert-base-v2":  # 12M parameters
            hidden_size = 768
        elif bert_model == "bert-base-cased": # 110M parameters
            hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        cont_reps, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids,return_dict=False)  #

        logits = self.cls_layer(self.dropout(pooler_output))

        return logits
        
        
 def set_seed(seed):
     """ Set all seeds to make results reproducible """
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     np.random.seed(seed)
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed)
 
 
 def evaluate_loss(net, device, criterion, dataloader):
     net.eval()
 
     mean_loss = 0
     count = 0
 
     with torch.no_grad():
         for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
             seq, attn_masks, token_type_ids, labels = \
                 seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
             logits = net(seq, attn_masks, token_type_ids)
             mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
             #mean_loss += criterion(logits.squeeze(-1), labels.type(torch.LongTensor).to(device)).item()
             count += 1
 
     return mean_loss / count

print("Creation of the models' folder...")
!mkdir ./file

def train_bert(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):

    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []

    scaler = GradScaler()

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):
            #print("1")

            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)

            with autocast():
                # Obtaining the logits from the model
                logits = net(seq, attn_masks, token_type_ids).to(device)  #

                # Computing loss
                #loss = criterion(logits.squeeze(-1), labels.type(torch.LongTensor).to(device))
                loss = criterion(logits.squeeze(-1), labels.float().to(device))
                loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

            # Backpropagating the gradients
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                scaler.step(opti)
                # Updates the scale for next iteration.
                scaler.update()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                opti.zero_grad()


            running_loss += loss.item()

            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, running_loss / print_every))

                running_loss = 0.0


        val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

        #net_copy2 = copy.deepcopy(net)
        #path_to_model_in_each_iter='./file/{}_lr_{}_val_loss_{}_ep_{}.pt'.format(bert_model, lr, round(best_loss, 5), best_ep)
        #torch.save(net_copy2.state_dict(), path_to_model_in_each_iter)
        #print("The model has been saved in {}".format(path_to_model))

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

    # Saving the model
    path_to_model='./file/{}_lr_{}_val_loss_{}_ep_{}.pt'.format(bert_model, lr, round(best_loss, 5), best_ep)
    torch.save(net_copy.state_dict(), path_to_model)
    print("The model has been saved in {}".format(path_to_model))

    del loss
    torch.cuda.empty_cache()

bert_model = "bert-base-cased"  # 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', ...
freeze_bert = False  # if True, freeze the encoder weights and only update the classification layer weights
maxlen =512 #512  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and elsef smaller, the input i is padded
bs = 16  # batch size
iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
lr = 2e-5  # learning rate
epochs = 5  # number of training epochs


#  Set all seeds to make reproducible results
set_seed(1234)

# Creating instances of training and validation set
print("Reading training data...")
train_set = CustomDataset(train_whole_data, maxlen, bert_model)
print("Reading validation data...")
val_set = CustomDataset(dev_whole_data, maxlen, bert_model)
# Creating instances of training and validation dataloaders
train_loader = DataLoader(train_set, batch_size=bs)
val_loader = DataLoader(val_set, batch_size=bs)
#print("pass")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = SentencePairClassifier(bert_model, freeze_bert=freeze_bert)

if torch.cuda.device_count() > 1:  # if multiple GPUs
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

net.to(device)

criterion = nn.BCEWithLogitsLoss()
opti = AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
num_warmup_steps = 0 # The number of steps for the warmup phase.
num_training_steps = epochs * len(train_loader)  # The total number of training steps
t_total = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

train_bert(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate)


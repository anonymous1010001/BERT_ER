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
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

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



print("Creation of the results' folder...")
!mkdir ./file/results

def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    #probs = nn.functional.softmax(logits)
    #print("logits",logits)
    probs = torch.sigmoid(logits)
    #print('probs',probs.shape)
    return probs.detach().cpu().numpy()

def test_prediction(net, device, dataloader, with_labels=True, result_file="results/output.txt"):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """
    net.eval()
    w = open(result_file, 'w')
    probs_all = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                #print("logits",logits.shape)
                probs = get_probs_from_logits(logits.squeeze(-1))
                probs_all += probs.tolist()
                #if len(probs_all)>32:
                 # break
        else:
            for seq, attn_masks, token_type_ids in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                #print("logits",logits)
                probs = get_probs_from_logits(logits.squeeze(-1))
                probs_all += probs.tolist()

    w.writelines(str(prob)+'\n' for prob in probs_all)
    w.close()
    
#store the model path from .py file of training        
path_to_model = ".."

path_to_output_file_test = './file/results/output.txt'


print("reading test data")
test_nli_set = CustomDataset(test_whole_data, maxlen, bert_model)
test_nli_loader = DataLoader(test_nli_set, batch_size=bs, num_workers=5)

# print("Reading test data...")
# test_set = CustomDataset(df_test, maxlen, bert_model)
# test_loader = DataLoader(test_set, batch_size=bs, num_workers=5)
print("test_nli_loader",test_nli_loader)
model = SentencePairClassifier(bert_model)
if torch.cuda.device_count() > 1:  # if multiple GPUs
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

print()
print("Loading the weights of the model...")
model.load_state_dict(torch.load(path_to_model))
model.to(device)


print("Predicting on test data...")
test_prediction(net=model, device=device, dataloader=test_nli_loader, with_labels=True,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                result_file=path_to_output_file_test)
print()
print("Predictions are available in : {}".format(path_to_output_file_test))

labels_test = test_whole_data['label']  # true labels
probs_test = pd.read_csv(path_to_output_file_test, header=None)[0]  # prediction probabilities
threshold = 0.5   # you can adjust this threshold for your own dataset
preds_test=(probs_test>=threshold).astype('uint8') # predicted labels using the above fixed threshold

claim_id_set=list(set(test_whole_data['Claim_id']))
org_label=list(test_whole_data['label'])
evi_id_list=list(test_whole_data['Evi_id'])
zip_pred=dict(zip(evi_id_list,preds_test))
zip_org=dict(zip(list(test_whole_data['Claim_id']),list(test_whole_data['label'])))

pred_test_dict={}
for i in zip_pred:
  #print(i)
  cl_=i.split("_")[0]
  lab_=zip_pred[i]
  if cl_ in pred_test_dict:
    pred_test_dict[cl_].append(lab_)
  else:
    pred_test_dict[cl_]=[]
    pred_test_dict[cl_].append(lab_)

pred_list_nw=[]
org_nw=[]
for i in pred_test_dict:
  mostFrequent = max(pred_test_dict[i], key=pred_test_dict[i].count)
  #print(mostFrequent)
  #cl_=i.split("_")[0]
  pred_list_nw.append(mostFrequent)
  org_nw.append(zip_org[int(i)])


# Compute the accuracy and F1 scores
f1=f1_score(org_nw, pred_list_nw, average='macro')
acc=accuracy_score(org_nw, pred_list_nw)  

print("Required F1-Score",f1)
print("Required Accuracy",acc

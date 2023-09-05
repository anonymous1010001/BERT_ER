import pandas as pd
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import pickle as pkl
from nltk.tokenize import word_tokenize
from transformers import pipeline, set_seed
import torch.utils.data as data
import torch
import numpy as np

def make_prompt(id_claim_pair):
    set_seed(42)
    output={}
    for i,j in enumerate(tqdm(id_claim_pair)):
      claim=id_claim_pair[j]
      part1= "Please provide a rationale either in favour of or against the assertion"
      part2="'{}'".format(claim)
      wh_sen=part1+' '+part2
      output[j]=wh_sen
    return output

def generate_synthetic_data(model,tokenizer,dataloader,num):
    train_evidence_gen={}
    for j,i in enumerate(tqdm(dataloader)):
      probability_val=[]
      encoding = tokenizer(i, padding=True, return_tensors='pt').to(device)
      with torch.no_grad():
        generated_outputs = model.to(device).generate(**encoding, return_dict_in_generate=True, output_scores=True, num_return_sequences=num, num_beams=2, max_length=52,max_new_tokens=50,top_k= 50,top_p= 0.95,do_sample=True,temperature=0.9)
        output_list=tokenizer.batch_decode(generated_outputs.sequences, skip_special_tokens=True)
        prob_output=generated_outputs.sequences_scores
        for k in prob_output:
          p=np.exp(k.cpu().numpy())
          probability_val.append(p)
        train_evidence_gen[j]={"claim":i, "generated_sen":output_list, "transition_val":probability_val}
    return train_evidence_gen
    

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create the training_data,valid_data and test_data
training_data = pd.read_csv(path_of_training_data)
valid_data = pd.read_csv(path_of_validation_data)
test_data = pd.read_csv(path_of_test_data)

X_train=list(training_data["CLAIM"])
X_train_id=list(training_data['ID'])
y_train=list(training_data["LABEL"])
X_test=list(test_data["CLAIM"])
y_test=list(test_data["LABEL"])
X_test_id=list(test_data['ID'])
X_val=list(valid_data["CLAIM"])
y_val=list(valid_data["LABEL"])
X_val_id=list(valid_data['ID'])

zip_new_tr=dict(zip(X_train_id,X_train))
zip_new_ts=dict(zip(X_test_id,X_test))
zip_new_val=dict(zip(X_val_id,X_val))
  
zip_new_tr_temp=make_prompt(zip_new_tr)
zip_new_val_temp=make_prompt(zip_new_val)
zip_new_ts_temp=make_prompt(zip_new_ts)

BATCH_SIZE = 8

train_iterator = data.DataLoader(list(zip_new_tr_temp.values()),
                                 shuffle=False,
                                 batch_size=BATCH_SIZE)

val_iterator = data.DataLoader(list(zip_new_val_temp.values()),
                                 shuffle=False,
                                 batch_size=BATCH_SIZE)

test_iterator = data.DataLoader(list(zip_new_ts_temp.values()),
                                 shuffle=False,
                                 batch_size=BATCH_SIZE)

#generated top 20 evidences for each claim
train_evidence_gen=generate_synthetic_data(model,tokenizer,dataloader=train_iterator,num=20)
val_evidence_gen=generate_synthetic_data(model,tokenizer,dataloader=val_iterator,num=20)
test_evidence_gen=generate_synthetic_data(model,tokenizer,dataloader=test_iterator,num=20)

fl_p=open(path_of_train_generated_text,"wb")
pickle.dump(train_evidence_gen,fl_p)
fl_p.close()

fl_p1=open(path_of_val_generated_text,"wb")
pickle.dump(val_evidence_gen,fl_p1)
fl_p1.close()

fl_p2=open(path_of_test_generated_text,"wb")
pickle.dump(test_evidence_gen,fl_p2)
fl_p2.close()

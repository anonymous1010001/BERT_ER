import pickle
import pandas as pd
import re
import random
from collections import OrderedDict
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def create_dict_from_qpp(give_tsv_file_location)
    file1 = open(give_tsv_file_location)
    file1_ls=file1.readlines()
    all_dict_id={}
    for lines in file1_ls:
        line_id_=lines.strip("\n").split("\t")[0]
        line_claim_=lines.strip("\n").split("\t")[1]
        line_string=lines.strip("\n").split("\t")[2:]
        if len(line_string)==0:
            random_line_string=="null"
        else:
            random_line_string=line_string
        all_dict_id[int(line_id_)]={"claim":line_claim_}
        all_dict_id[int(line_id_)].update({"evidence":random_line_string})
    return all_dict_id

def create_list_for_csv(give_the_dict_from_tsv,id_lab_test):
    claim_all=[]
    evi_all=[]
    evi_id_all=[]
    id_all=[]
    label_all=[]
    for i in give_the_dict_from_tsv:
      spitted_evi=give_the_dict_from_tsv[i]['evidence'][:5]
      cl=give_the_dict_from_tsv[i]['claim']
      #print(spitted_evi)
      for k,j in enumerate(spitted_evi):
      #   print(j.split('/t'))
        sen_id=str(i)+"_"+str(k)
        #print(sen_id)
        id_all.append(i)
        evi_all.append(j)
        evi_id_all.append(sen_id)
        claim_all.append(cl)
        label_all.append(id_lab_test[i])
    return id_all, claim_all, evi_id_all, evi_all, label_all

## create a basic csv file from the FEVER jsonl files, where the columns are "id","Claim","label".
training_data = pd.read_csv(train_csv_file)
test_data = pd.read_csv(test_csv_file)

X_tr=training_data["CLAIM"]
y_tr=training_data["LABEL"]
X_tr_ID=training_data["ID"]
X_test=test_data["CLAIM"]
y_test=test_data["LABEL"]
X_test_ID=test_data["ID"]

cl_id_pair_ts=dict(zip(X_test,X_test_ID))
cl_lb_pair_ts=dict(zip(X_test,y_test))
cl_id_pair_tr=dict(zip(X_tr,X_tr_ID))
cl_lb_pair_tr=dict(zip(X_tr,y_tr))

id_lab_tr=dict(zip(X_tr_ID,y_tr))
id_cl_tr=dict(zip(X_tr_ID,X_tr))
id_lab_test=dict(zip(X_test_ID,y_test))
id_cl_test=dict(zip(X_test_ID,X_test))

#create the dictionary from .tsv file
all_dict_id_tr=create_dict_from_qpp(give_tsv_file_location="./train_retrieved.tsv")
all_dict_id_ts=create_dict_from_qpp(give_tsv_file_location="./test_retrieved.tsv")

id_all, claim_all, evi_id_all, evi_all, label_all=create_list_for_csv(give_the_dict_from_tsv=all_dict_id_tr,id_lab_test=id_lab_tr)
id_all_ts, claim_all_ts, evi_id_all_ts, evi_all_ts, label_all_ts=create_list_for_csv(give_the_dict_from_tsv=all_dict_id_tr,id_lab_test=id_lab_test)

df_tr = pd.DataFrame(list(zip(id_all, claim_all, evi_id_all, evi_all, label_all)),columns =['Claim_id', 'Claim', 'evi_id','Evidence', 'label'])
df_ts = pd.DataFrame(list(zip(id_all_ts, claim_all_ts, evi_id_all_ts, evi_all_ts, label_all_ts)),columns =['Claim_id', 'Claim', 'evi_id','Evidence', 'label'])

dev_whole_data_cl_id=list(df_tr['Claim_id'])
dev_whole_data_cl=list(df_tr['Claim'])
dev_whole_data_evi_id=list(df_tr['evi_id'])
dev_whole_data_evi=list(df_tr['Evidence'])
dev_whole_label=list(df_tr['label'])

zip_list_train_whole=list(zip(dev_whole_data_cl_id,dev_whole_data_cl,dev_whole_data_evi_id, dev_whole_data_evi, dev_whole_label))
train,dev = train_test_split(zip_list_train_whole, test_size=0.1, random_state=42)
zip_df_train = pd.DataFrame(train,columns=['Claim_id', 'Claim', 'evi_id','Evidence', 'label'])
zip_df_val = pd.DataFrame(dev,columns=['Claim_id', 'Claim', 'evi_id','Evidence', 'label'])

zip_df_train.to_csv("./train_data_wiki_cl_evi_5.csv")
zip_df_val.to_csv("./val_data_wiki_cl_evi_5.csv")
df_ts.to_csv("./test_data_wiki_cl_evi_5.csv")

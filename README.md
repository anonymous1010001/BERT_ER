# BERT_ER
This repository contains source code for LLM generated synthetic evidence based Fact Verification Task.

## Prepare Dataset 
### Dataset corresponding BERT_SD
Run `generated_sentence_creation.py` to generate the LLM generated synthetic data and then convert that into .csv format using `generation_pickle_data_to_csv_convert.py` file.

### Dataset corresponding BERT_FSD
First create a `.csv` file including `claim_id`, `claim`,`label`, using the annotated FEVER .jsonl files.<br> 
Then run `generated_sentence_creation.py` to generate the LLM generated synthetic data. <br> 
Using these two sets of .csv files(train, test and validation) run the `create_filtered_data.py` dataset required in BERT_FSD model.

### Dataset corresponding BERT_ER
First create a `.csv` file including `claim_id`, `claim`,`label`, using the annotated FEVER .jsonl files.<be>
Then run `wiki_chunk_wise_data.py` for the required dataset of BERT_ER model.

## Training and inferencing individual models
For training run `training_with_LLM_generated_synthetic_data.py`. <br> After training, you can find the best checkpoint on the dev set according to the evaluation results. For this use `prediction.py`.

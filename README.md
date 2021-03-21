# BERTrip
BERTrip: Language model for trips
Use python 3.

## Prepare data
- Get trajectories from mongodb
- `python trips_to_h3.py` to map geocordinates into hexagons
- `cut -d',' -f4 data/processed/trips_h3_10.txt > data/processed/trips_h3_10_textonly.txt`
- tar this file and copy to scdev5:/var/www/sabbar
  
## Pretrain BERTrip from scratch (Google colab/TPU):
- Use this collab https://colab.research.google.com/drive/1thO6_T063xOR2o6_yCdVxKmyQYsTeG3v#scrollTo=RiFH_9Lbze5f

## Copy data to local machine:
* `~/tools/google-cloud-sdk/bin/gsutil -m cp   "gs://bertrip/bert_model/bert_config.json"   "gs://bertrip/bert_model/model.ckpt-92500.data-00000-of-00001"   "gs://bertrip/bert_model/model.ckpt-92500.index"   "gs://bertrip/bert_model/model.ckpt-92500.meta"   "gs://bertrip/bert_model/vocab.txt"   .`

## Convert tensorflow checkpoints into PyTorch save file as follows:
* create a `config.json` from `bert_config.json` by adding: `"model_type": "bert"`
```json
{
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "model_type": "bert",
  "max_position_embeddings": 512,
  "num_attention_heads": 12, 
  "num_hidden_layers": 12, 
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12, 
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 32000
}
```
* export BERT_BASE_DIR='models'
* transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin

## Run unmasker locally:
* create virtualenv: python3 -m venv venv
* activate it: . venv/bin/activate
* install packages: pip install -r requirements
* run example: python hf_unmasker.py

# Finetune a downstream task for ETA prediction
The objective here is: given a trajectory, find its ETA.
At first, we finetune on trajectories starting all at the same hour, e.g., 8am.
We can finetune other models for other hours.
* check code in (not working 100%): python hf_train_eta.py 


# Deployment on TorchServer
- Use Docker
-Go to: cd ~/projects/2021/

## Generate Archive
### Start torchserver:
Make sure you add create a config file: config.properties with the following content:
`
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=32
job_queue_size=1000
model_store=/home/model-server/model-store
install_py_dep_per_model=true
`
Last line is important to allow installation of python packages in requirements.txt

Then run (with link to new config file):
docker run --rm -it -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/model_store/config.properties:/home/model-server/config.properties $(pwd)/model_store:/home/model-server/model-store  pytorch/torchserve:latest-cpu

### access the container
docker exec -it mar bash

### Create archive
torch-model-archiver --model-name "bert" --version 1.0 --serialized-file ./model-store/pytorch_model.bin --extra-files "./model-store/config.json,./model-store/vocab.txt" --handler "./model-store/transformers_fillmask_torchserve_handler.py" --requirements-file "./model-store/requirements.txt"

## Run torchserve
docker run --rm --shm-size=1g         --ulimit memlock=-1         --ulimit stack=67108864         -p8080:8080         -p8081:8081         -p8082:8082         -p7070:7070         -p7071:7071    --name torchserve     --mount type=bind,source=$(pwd)/model_store,target=/home/model-server/model-store  pytorch/torchserve:latest-cpu torchserve --model-store /home/model-server/model-store --models bert.mar 


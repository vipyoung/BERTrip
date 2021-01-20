# BERTrip
BERTrip: Language model for trips


## Prepare data
- Get trajectories from mongodb
- `python trips_to_h3.py` to map geocordinates into hexagons
- `cut -d',' -f4 data/processed/trips_h3_10.txt > data/processed/trips_h3_10_textonly.txt`
- tar this file and copy to scdev5:/var/www/sabbar
  
## Pretrain BERTrip from scratch (Google colab/TPU):
- Use this collab https://colab.research.google.com/drive/1thO6_T063xOR2o6_yCdVxKmyQYsTeG3v#scrollTo=RiFH_9Lbze5f

## Copy data to local machine:
- `~/tools/google-cloud-sdk/bin/gsutil -m cp   "gs://bertrip/bert_model/bert_config.json"   "gs://bertrip/bert_model/model.ckpt-92500.data-00000-of-00001"   "gs://bertrip/bert_model/model.ckpt-92500.index"   "gs://bertrip/bert_model/model.ckpt-92500.meta"   "gs://bertrip/bert_model/vocab.txt"   .`

- Convert tensorflow checkpoints into PyTorch save file as follows:
-- export BERT_BASE_DIR='models/92500'
-- transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin

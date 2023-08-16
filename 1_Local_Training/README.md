# If you are using deepspeed 
## Set up the path in ds_train_finetune.sh
```bash
  --train_file /path/to/the/train/file \
  --test_file /path/to/the/test/file \
  --model_name_or_path /path/to/the/model/weight \
  --output_dir /path/to/the/output/adgen-chatglm-6b-ft-$LR \
```
## Start local finetune with Deepspeed

```bash
cd 1_Local_Training
bash ds_train_finetune.sh
```

# If you are using Ptuning v2

## Set up the path in

```bash
  --train_file /path/to/the/train/file \
  --validation_file /path/to/the/val/file \
  --model_name_or_path  /path/to/the/model/weight \
  --output_dir /path/to/the/output/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
```

## Start local finetune with Ptuning v2

```bash
cd 1_Local_Training
bash ptuning.sh
```


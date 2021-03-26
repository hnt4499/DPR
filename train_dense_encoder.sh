CUDA_VISIBLE_DEVICES=5,7 python -m torch.distributed.launch --master_port=4137 --nproc_per_node=2 \
    train_dense_encoder.py \
    encoder=hf_bert_single_model \
    train=biencoder_nq_single_model \
    train_datasets=[nq_train,nq_train_hn1] \
    dev_datasets=[nq_dev] \
    output_dir=checkpoints/
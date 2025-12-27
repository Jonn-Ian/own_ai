# Convert Dataset
    Download and convert a Hugging Face dataset into JSONL:
    
    commands:
    
    python train.py convert_hf --dataset daily_dialog --out_dir ./raw_data
    

# Preprocess 
    Deduplicate, split, and shard:

    commands:

    python train.py preprocess --input_dir ./raw_data --out_dir ./processed --val_ratio 0.02 --shard_size 200000

# Build Tokens

    Train SentencePiece and tokenize:

    commands:

    python train.py build_tokens --data_dir ./processed --save_dir ./tokens --sp_prefix my_spm --vocab_size 32000 --max_seq_len 1024 --stride 512

# Train
    Run supervised fine‑tuning (SFT):

    commands:

    python train.py train --tokens_dir ./tokens --save_dir ./checkpoints --batch_size 4 --grad_accum_steps 8 --total_steps 100000 --use_checkpoint

### For distributed training ###
    commands:
    
    torchrun --nproc_per_node=4 train.py train --tokens_dir ./tokens --save_dir ./checkpoints
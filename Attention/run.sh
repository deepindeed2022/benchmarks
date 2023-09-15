TAG=nvcr.io/nvidia/pytorch:23.04-py3_sdpa

docker build -f ./Dockerfile -t $TAG .

for num_heads in 8 16 32 40; do
    for batch_size in 1 2 4 8 16 32 64 128; do
        for seq_len in 256 512 1024 2048 4096; do
            docker run  --gpus all -v ./:/workspace $TAG \
                            python3 -u test_attention.py --batch_size $batch_size --seq_len $seq_len --num_heads $num_heads --emb_dim 128
        done
    done
done
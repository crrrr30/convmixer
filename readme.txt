torchrun --nproc_per_node=8 train.py \
    --model asmlp_base_patch4_shift5_224 \
    -b 128 \
    -j 10 \
    --opt adamw \
    --epochs 150 \
    --sched onecycle \
    --amp \
    --input-size 3 224 224 \
    --lr 0.01 \
    --aa rand-m9-mstd0.5-inc1 \
    --cutmix 0.5 \
    --mixup 0.5 \
    --reprob 0.25 \
    --remode pixel \
    --num-classes 1000 \
    --warmup-epochs 0 \
    --opt-eps=1e-3 \
    --clip-grad 1.0 \
    --resume output/train/20230613-163055-asmlp_base_patch4_shift5_224-224/checkpoint-56.pth.tar \
    &> logs.out

torchrun --nproc_per_node=8 train.py \
    --model asmlp_base_patch4_shift5_224 \
    -b 128 \
    -j 10 \
    --opt adamw \
    --epochs 300 \
    --warmup-epochs 20 \
    --weight-decay 0.05 \
    --lr 0.001 \
    --warmup-lr 0.000001 \
    --min-lr 0.00001 \
    --sched cosine \
    --decay-epochs 30 \
    --decay-rate 0.1 \
    --amp \
    --input-size 3 224 224 \
    --aa rand-m9-mstd0.5-inc1 \
    --cutmix 1.0 \
    --mixup 0.8 \
    --reprob 0.25 \
    --remode pixel \
    --num-classes 1000 \
    --opt-eps=1e-3 \
    --clip-grad 5.0 \
    &> logs.out

pip install datasets einops cupy-cuda11x tabulate
git clone https://github.com/crrrr30/convmixer -b arrow

wget https://github.com/Backblaze/B2_Command_Line_Tool/releases/latest/download/b2-linux
chmod +x b2-linux
./b2-linux authorize-account
./b2-linux sync b2://jcresearch/imagenet-1k/default/1.0.0/a1e9bfc56c3a7350165007d1176b15e9128fcaf9ab972147840529aed3ae52bc imagenet-1k

apt update && apt install -y vim bmon less tmux g++ && bmon

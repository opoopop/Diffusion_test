# Command helper



## Linear schedule test on unconditional  cifar-10

```bash
# model train
export OPENAI_LOGDIR=~/improved-diffusion-main/ML_project/linear_schedule

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --save_interval 50000"


python scripts/image_train.py --data_dir /home/chunjie/improved-diffusion-main/datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

# sample
python scripts/image_sample.py \
--model_path /home/chunjie/improved-diffusion-main/ML_project/linear_schedule/ema_0.9999_500000.pt \
--num_samples 50000 \
--use_ddim True \
--timestep_respacing ddim500 \
--batch_size 64 \
$MODEL_FLAGS $DIFFUSION_FLAGS

```

## Cosine schedule test on unconditional  cifar-10

```bash
# training 
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

# sample
python scripts/image_sample_uncond.py \
--model_path /home/chunjielu/improved-diffusion-main/models/ema_0.9999_500000.pt \
--num_samples 50000 \
--use_ddim True \
--timestep_respacing ddim500 \
--batch_size 64 \
$MODEL_FLAGS $DIFFUSION_FLAGS
```

## DDIM sample with different steps

```bash
# sample
# replace ddim500 -> ddim25/ddim50/ddim100/ddim500/ddim1000
python scripts/image_sample_uncond.py \
--model_path /home/chunjielu/improved-diffusion-main/models/ema_0.9999_500000.pt \
--num_samples 50000 \
--use_ddim True \
--timestep_respacing ddim500 \
--batch_size 64 \
$MODEL_FLAGS $DIFFUSION_FLAGS
```

## Guided Diffusion

```bash
# imagenet 64x64
SAMPLE_FLAGS="--batch_size 8 --num_samples 10000 --timestep_respacing ddim100 --use_ddim True"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"


python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt --classifier_depth 4 --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS


# 256x256
SAMPLE_FLAGS="--batch_size 1 --num_samples 5 --timestep_respacing ddim100 --use_ddim True"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

python classifier_sample.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS

#  style fusion
SAMPLE_FLAGS="--batch_size 1 --num_samples 5 --timestep_respacing ddim100 --use_ddim True"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

python fusion_style_sample.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS

```


export CUDA_VISIBLE_DEVICES=2
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

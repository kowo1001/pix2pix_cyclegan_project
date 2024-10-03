
export CUDA_VISIBLE_DEVICES=3
set -ex
python train.py --dataroot ./datasets/maps --name maps_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0

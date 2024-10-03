export CUDA_VISIBLE_DEVICES=2
set -ex
python test_pix2pix.py --dataroot ./datasets/maps --name maps_pix2pix --model pix2pix --netG unet_256 --direction AtoB --dataset_mode aligned --norm batch

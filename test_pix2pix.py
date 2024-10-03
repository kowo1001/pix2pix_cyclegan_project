"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
from sklearn.metrics import confusion_matrix
import glob
from PIL import Image


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def calculate_class_iou(ground_truth, prediction):
    intersection = np.sum(np.where(np.all(ground_truth == prediction, axis=2), 1, 0))
    ground_truth_sum = np.sum(np.all(ground_truth > 0, axis=2))
    prediction_sum = np.sum(np.all(prediction > 0, axis=2))
    iou = intersection / (ground_truth_sum + prediction_sum - intersection)
    return iou

def calculate_per_class_accuracy(ground_truth, prediction):
    conf_matrix = confusion_matrix(np.argmax(ground_truth, axis=2), np.argmax(prediction, axis=2))
    accuracy = np.sum(conf_matrix.diagonal()) / conf_matrix.sum()
    return accuracy

def calculate_per_pixel_accuracy(ground_truth, prediction):
    accuracy = np.mean(np.all(ground_truth == prediction, axis=2))
    return accuracy


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

    # 생성된 이미지 디렉토리 경로 설정
    directory_path = '/home/jwjang/project/cyclegan_pix2pix_pr/2step/CycleGAN-and-pix2pix/results/maps_pix2pix/test_latest/images'
    # 원본 이미지 디렉토리 경로 설정
    org_directory_path = '/home/jwjang/project/cyclegan_pix2pix_pr/2step/CycleGAN-and-pix2pix/datasets/maps/test'
    print('org_directory_path: ', org_directory_path)

    # 해당 디렉토리에 있는 모든 PNG 파일 가져오기
    pred_png_files = glob.glob(directory_path + '/57_DEM_fake_B.png')
    org_png_files = glob.glob(org_directory_path + '/*.bmp')

    print('png_files 개수: ', len(pred_png_files))
    print('org_png_files 개수: ', len(org_png_files))

    # gt_png_files = [png_files[i].replace('.png', '_gt.png') for i in range(len(png_files))]

    arg_size = (512, 512)

    generated_images_resized = []
    for img_path in org_png_files:
        img = Image.open(img_path)
        Image.ANTIALIAS = Image.LANCZOS
        resized_img = img.resize(arg_size, Image.ANTIALIAS)
        generated_images_resized.append(np.array(resized_img))

    # ground_truth_images = [np.array(Image.open(org_png_files[i])) for i in range(len(org_png_files))]
    generated_images = [np.array(Image.open(pred_png_files[i])) for i in range(len(pred_png_files))]

    print('generated_images 개수: ', len(generated_images))
    print('ground_truth_images 개수: ', len(generated_images_resized))
    print('generated_images_resized shape: ', generated_images[0].shape) #  (512, 512, 3)
    print('ground_truth_images shape: ', generated_images_resized[0].shape) # (512, 512)

    generated_images_resized[0] = np.expand_dims(generated_images_resized[0], axis=2) # 차원 확장
    print('ground_truth_images_expanded shape: ',generated_images_resized[0].shape) #  (512, 512, 1) # 왜 차원 확장하면 값이 ?

    for ground_truth, generated in zip(generated_images_resized, generated_images):
        print('*ground_truth_expanded: ', ground_truth[0].shape) # (512,)
        print('*generated_expanded: ', generated[0].shape) # (512, 3)
        # ground_truth = np.expand_dims(ground_truth, axis=2) # 차원 확장
        print('***ground_truth_expanded: ', ground_truth[0].shape) # (512, 1)
        print('***generated_expanded: ', generated[0].shape) # (512, 3)
        # plt.imshow(ground_truth-generated)

    class_iou = 0
    per_class_accuracy = 0
    per_pixel_accuracy = 0
    for ground_truth, generated in zip(generated_images_resized, generated_images):
        # ground_truth = np.expand_dims(ground_truth, axis=2) # 차원 확장
        class_iou += calculate_class_iou(ground_truth, generated)
        # per_class_accuracy += calculate_per_class_accuracy(ground_truth, generated)
        per_pixel_accuracy += calculate_per_pixel_accuracy(ground_truth, generated)

    # 평균 평가 지표 계산
    average_class_iou = class_iou / len(generated_images_resized)
    # average_per_class_accuracy = per_class_accuracy / len(generated_images_resized)
    average_per_pixel_accuracy = per_pixel_accuracy / len(generated_images_resized)

    print("Average Class IOU:", average_class_iou)
    # print("Average Per-class Accuracy:", average_per_class_accuracy)
    print("Average Per-pixel Accuracy:", average_per_pixel_accuracy)

    webpage.save()  # save the HTML

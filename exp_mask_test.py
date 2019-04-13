import torch
from torch.autograd import Variable

from scipy.misc import imresize
from imageio import imread, imsave
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import PoseExpNet
from inverse_warp import pose_vec2mat
from utils import tensor2array
from PIL import Image

parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--sequences", default=['2011_09_26_drive_0002_02'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    args = parser.parse_args()
    from kitti_eval.exp_mask_utils import test_framework_KITTI as test_framework

    weights = torch.load(args.pretrained_posenet, map_location=lambda storage, loc: storage)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=True).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))

    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']

        h,w,_ = imgs[0].shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]

        imgs = [np.transpose(img, (2,0,1)) for img in imgs]

        ref_imgs = []
        for i, img in enumerate(imgs):
            img = torch.from_numpy(img).unsqueeze(0)
            img = ((img/255 - 0.5)/0.5).to(device)
            if i == len(imgs)//2:
                tgt_img = img
            else:
                ref_imgs.append(img)

        exp_mask, poses = pose_net(tgt_img, ref_imgs)

        # print('This is the maskp')
        print(exp_mask.data.size())

        args.output_disp = True

        if args.output_disp:
            # disp_ = exp_mask.data[:,2,:,:].reshape(1,128,416)
            # print(disp_)
            # disp = (255*tensor2array(disp_, max_value=10, colormap='bone')).astype(np.uint8)
            max_value = exp_mask.data.max().item()
            array_1 = exp_mask.data[:,0,:,:].squeeze().numpy()
            print(array_1)
            array_1 = (255*array_1).astype(np.uint8)
            print(array_1)
            print(np.min(array_1))

            imsave(output_dir/'{}_disp{}'.format(j, '.png'), array_1)

if __name__ == '__main__':
    main()

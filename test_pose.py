import csv
import numpy as np
import os 
import torch

from torch.autograd import Variable
from skimage.transform import resize as imresize
from path import Path
from tqdm import tqdm
from natsort import natsorted
from models import PoseExpNet, SemPoseExpNet
from utils.inverse_warp import pose_vec2mat
from imageio import imread
from utils.common import array2tensor

# parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
# parser.add_argument("--img-height", default=128, type=int, help="Image height")
# parser.add_argument("--img-width", default=416, type=int, help="Image width")
# parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
# parser.add_argument("--min-depth", default=1e-3)
# parser.add_argument("--max-depth", default=80)

# parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
# parser.add_argument("--sequences", default=['09'], type=str, nargs='*', help="sequences to test")
# parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
# parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
# parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_ref_images(files, index, sequence = [-1, 1]):
    # source images
    tensor_imgs = []
    for s in sequence:
        simg_ = imread(files[index+s])
        simg = array2tensor(simg_).to(device)
        tensor_imgs.append(simg)
    return tensor_imgs

@torch.no_grad()
def main(args):
    # Load pose network
    if args.with_semantics:
        pose_net = SemPoseExpNet(output_exp=args.with_mask).to(device)
    else:
        pose_net = PoseExpNet(output_exp=args.with_mask).to(device)
    
    weights = torch.load(args.pretrained)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    
    # Path
    dataset_dir = Path(args.dataset_dir)
    base_output_dir = Path(args.output_dir)
    base_output_dir.makedirs_p()
    
    # with open(base_output_dir/args.error_summary, 'w') as csvfile:
    #     writer = csv.writer(csvfile, delimiter='\t')
    #     writer.writerow(error_names)

    # with open(base_output_dir/args.error_full, 'w') as csvfile:
    #     writer = csv.writer(csvfile, delimiter='\t')
    #     writer.writerow(error_names)
    
    dirs = [dataset_dir/d for d in os.listdir(dataset_dir)]
    ATE, RE = [], []
    for dir_ in dirs:
        print(f"on dir {dir_}")
        
        episodes = os.listdir(dir_)
    
        for i, ep in enumerate(episodes):
            if i > args.num_episodes:
                break
            print(f"on episode {ep}")
            
            rgb_indir = dir_/ep/'rgb'
            rgb_files = natsorted(rgb_indir.walkfiles('*.png'))
            
            sem_indir = dir_/ep/'semantics'
            sem_files = natsorted(sem_indir.walkfiles('*.png'))

            poses_file = dir_/ep/'poses.npy'
            gt_poses = np.load(poses_file)
            pred_poses = []
            for i in range(1, len(rgb_files)-1):   
                             
                # target image
                out_img = imread(rgb_files[i])
                tgt_img = array2tensor(out_img).to(device)
                
                # semantic images
                sem_out_img = imread(sem_files[i])
                sem_tgt_img = array2tensor(sem_out_img).to(device)

                # source images
                ref_imgs = get_ref_images(rgb_files, i)
                sem_ref_imgs = get_ref_images(sem_files, i)
                
                # predicted depth
                if args.with_semantics:
                    _, pred_pose = pose_net(
                        tgt_img, ref_imgs, sem_tgt_img, sem_ref_imgs)
                else:
                    _, pred_pose = pose_net(tgt_img, ref_imgs)    
                
                pred_poses.append(pred_pose[0][1])
            
            inv_gt_poses = pose_vec2mat(torch.from_numpy(gt_poses))[:-1]
            inv_gt_poses = inv_gt_poses.detach().cpu().numpy()
            inv_pred_poses = pose_vec2mat(torch.stack(pred_poses))
            inv_pred_poses = inv_pred_poses.detach().cpu().numpy()
            
            gt_rot_mats = np.linalg.inv(inv_gt_poses[:,:,:3])
            gt_tr_vecs = -gt_rot_mats @ inv_gt_poses[:,:,-1:]
            gt_tf_matrices = np.concatenate([gt_rot_mats, gt_tr_vecs], axis=-1)
            
            gt_fst_inv_tf = inv_gt_poses[0]
            gt_final_poses = gt_fst_inv_tf[:,:3] @ gt_tf_matrices
            gt_final_poses[:,:,-1:] += gt_fst_inv_tf[:,-1:]
            
            pred_rot_mats = np.linalg.inv(inv_pred_poses[:,:,:3])
            pred_tr_vecs = -pred_rot_mats @ inv_pred_poses[:,:,-1:]
            pred_tf_matrices = np.concatenate([pred_rot_mats, pred_tr_vecs], axis=-1)
            
            pred_fst_inv_tf = inv_pred_poses[0]
            pred_final_poses = pred_fst_inv_tf[:,:3] @ pred_tf_matrices
            pred_final_poses[:,:,-1:] += pred_fst_inv_tf[:,-1:]

            ate, re = compute_pose_error(gt_final_poses, pred_final_poses)
            ATE.append(ate)
            RE.append(re)

    ATE = np.stack(ATE)
    RE = np.stack(RE)

    error_names = ['ATE','RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(ATE.mean(), RE.mean()))
    print("std \t {:10.4f}, {:10.4f}".format(ATE.std(), RE.std()))

def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pose test script for the MP3D Sfm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # I/O
    parser.add_argument("--pretrained", required=False, type=str, 
        help="pretrained PoseNet path")
    parser.add_argument("--dataset-dir", default='./data/mp3d_sfm/val_unseen', 
        type=str, help="Dataset directory")
    parser.add_argument("--output-dir", default='out', type=str,  
        help="Output directory")
    parser.add_argument("--error-summary", default='error_summary.csv', type=str,
        help='file to save computed errors')
    parser.add_argument("--error-full", default='error_full.csv', type=str,
        help='file to save computed errors')
    
    # Other parameters
    parser.add_argument("--with-semantics", action='store_true', 
        help='Use SemPoseExpNet or PoseExpNet')
    parser.add_argument("--with-mask", action='store_true', 
        help='Use SemPoseExpNet or PoseExpNet')
    parser.add_argument("--num-episodes", default=10, type=int,
        help="number of episodes to run")
    
    args = parser.parse_args()
    main(args)
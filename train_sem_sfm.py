import time
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import utils.custom_transforms as custom_transforms
import models

from datasets.sequence_mp3d import SequenceFolderWithSemantics
from utils.common import (
    tensor2array, save_checkpoint, save_path_formatter, 
    log_output_tensorboard_semantic)
from utils.loss_functions import (photometric_reconstruction_loss, smooth_loss,
    explainability_loss, semantic_reconstruction_loss)
from utils.logger import TermLogger, AverageMeter

from tensorboardX import SummaryWriter

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(args):
    global best_error, n_iter, device
    
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    
    torch.manual_seed(args.seed)
    
    if args.evaluate:
        args.epochs = 0

    tb_writer = SummaryWriter(args.save_path)
    
    # Data loading code
    print("=> fetching scenes in '{}'".format(args.data))
    
    normalize = custom_transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    train_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(), normalize])
    
    train_set = SequenceFolderWithSemantics(
        args.data, transform=train_transform, seed=args.seed, train=True, 
        sequence_length=args.sequence_length)
    
    valid_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(), normalize])

    val_set = SequenceFolderWithSemantics(
        args.data, transform=valid_transform, seed=args.seed, train=False, 
        sequence_length=args.sequence_length)
    
    print('\t{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('\t{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, pin_memory=args.pin_memory)
    
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.pin_memory)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    
    disp_net = models.SemDispNetS().to(device)
  
    output_exp = args.mask_loss_weight > 0
    if not output_exp:
        print("=> no mask loss, PoseExpnet will only output pose")
        
    pose_exp_net = models.SemPoseExpNet(
        nb_ref_imgs=args.sequence_length - 1, output_exp=output_exp).to(device)

    if args.pretrained_exp_pose:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_exp_pose)
        pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        pose_exp_net.init_weights()

    if args.pretrained_disp:
        print("=> using pre-trained weights for Dispnet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'])
    else:
        disp_net.init_weights()

    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    pose_exp_net = torch.nn.DataParallel(pose_exp_net)

    print('=> setting adam solver')

    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_exp_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(
        optim_params, betas=(args.momentum, args.beta),
        weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([
            'train_loss', 'photo_loss', 'explainability_loss', 'smooth_loss', 
            'semantic_loss'])

    logger = TermLogger(
        n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), 
        valid_size=len(val_loader))
    logger.epoch_bar.start()

    if args.pretrained_disp or args.evaluate:
        logger.reset_valid_bar()
        errors, error_names = validate(
            args, val_loader, disp_net, pose_exp_net, 0, logger, tb_writer)
        
        for error, name in zip(errors, error_names):
            tb_writer.add_scalar(name, error, 0)
        
        error_string = (
            ', '.join('{} : {:.3f}'.format(name, error) 
            for name, error in zip(error_names[2:9], errors[2:9])))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(
            args, train_loader, disp_net, pose_exp_net, optimizer, 
            args.epoch_size, logger, tb_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        errors, error_names = validate(
            args, val_loader, disp_net, pose_exp_net, epoch, logger, tb_writer)
        
        error_string = (
            ', '.join('{} : {:.3f}'.format(name, error) 
            for name, error in zip(error_names[2:9], errors[2:9])))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            tb_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's 
        # performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, 
            {'epoch': epoch + 1, 'state_dict': disp_net.module.state_dict()}, 
            {'epoch': epoch + 1, 'state_dict': pose_exp_net.module.state_dict()}, 
            is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_exp_net, optimizer, epoch_size, 
    logger, tb_writer):
    global n_iter, device
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    
    w1 = args.photo_loss_weight
    w2 = args.mask_loss_weight
    w3 = args.smooth_loss_weight
    w4 = args.semantic_loss_weight
    
    # switch to train mode
    disp_net.train()
    pose_exp_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, batch in enumerate(train_loader):
        target, target_sem, source, source_sem, intrinsics, intrinsics_inv = batch 
    
        log_losses = i > 0 and n_iter % args.print_freq == 0
        log_output = (
            args.training_output_freq > 0 and n_iter % args.training_output_freq == 0)

        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(device)
        target_sem = target_sem.to(device)
        
        source = [img.to(device) for img in source]
        source_sem = [img.to(device) for img in source_sem]
        
        intrinsics = intrinsics.to(device)

        # predict disparitites
        disparities = disp_net(target, target_sem)
        depth = [1/disp for disp in disparities]
       
        # predict pose and explainability mask
        explainability_mask, pose = pose_exp_net(
            target, source, target_sem, source_sem)

        # compute loss
        loss_1, warped, diff = photometric_reconstruction_loss(
            target, source, intrinsics, depth, explainability_mask, pose,
            args.rotation_mode, args.padding_mode)
        
        loss_2 = 0
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask)
        
        loss_3 = smooth_loss(depth)

        loss_4 = 0
        if w4 > 0:
            loss_4, warped_sem, diff_sem = semantic_reconstruction_loss(
                target_sem, source_sem, intrinsics, depth, explainability_mask, 
                pose, args.rotation_mode, args.padding_mode)
        
        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3 + w4 * loss_4

        if log_losses:
            tb_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            
            if w2 > 0:
                tb_writer.add_scalar('explanability_loss', loss_2.item(), n_iter)
                
            tb_writer.add_scalar('disparity_smoothness_loss', loss_3.item(), n_iter)
            
            if w4 > 0:
                tb_writer.add_scalar('semantic_error', loss_4.item(), n_iter)
                
            tb_writer.add_scalar('total_loss', loss.item(), n_iter)

        if log_output:
            tb_writer.add_image('train Input', tensor2array(target[0]), n_iter)
            tb_writer.add_image('semantic Input', tensor2array(target_sem[0]), n_iter)
            for k, scaled_maps in enumerate(
                zip(depth, disparities, warped, diff, warped_sem, diff_sem, 
                    explainability_mask)):
                
                log_output_tensorboard_semantic(
                    tb_writer, "train", 0, " {}".format(k), n_iter, *scaled_maps)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([
                loss.item(), loss_1.item(), loss_2.item() if w2 > 0 else 0, 
                loss_3.item(), loss_4.item() if w4 > 0 else 0, loss_4.item()])
        
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write(
                'Train: Time {} Data {} Loss {}'.format(batch_time, data_time, 
                losses))
            
        if i >= epoch_size - 1:
            break

        n_iter += 1
        
    return losses.avg[0]

@torch.no_grad()
def validate(
    args, val_loader, disp_net, pose_exp_net, epoch, logger, tb_writer, 
    sample_nb_to_log=3):
    global device
    
    batch_time = AverageMeter()
    losses = AverageMeter(i=3, precision=4)
    log_outputs = sample_nb_to_log > 0
    
    # Output the logs throughout the whole dataset
    batches_to_log = list(
        np.linspace(0, len(val_loader), sample_nb_to_log).astype(int))
    
    w1 = args.photo_loss_weight
    w2 = args.mask_loss_weight
    w3 = args.smooth_loss_weight
    w4 = args.semantic_loss_weight
    
    poses = np.zeros(
        ((len(val_loader)-1) * args.batch_size * (args.sequence_length-1), 6))
    disp_values = np.zeros(((len(val_loader)-1) * args.batch_size * 3))

    # switch to evaluate mode
    disp_net.eval()
    pose_exp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, batch in enumerate(val_loader):
            
        target, target_sem, source, source_sem, intrinsics, intrinsics_inv = batch
        
        target = target.to(device)
        target_sem = target_sem.to(device)
        
        source = [img.to(device) for img in source]
        source_sem = [img.to(device) for img in source_sem]
        
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)
        
        # predict disparitites
        disp = disp_net(target, target_sem)
        depth = 1 / disp
       
        # predict pose and explainability mask
        explainability_mask, pose = pose_exp_net(
            target, source, target_sem, source_sem)
        
        # compute loss
        loss_1, warped, diff = photometric_reconstruction_loss(
            target, source, intrinsics, depth, explainability_mask, pose,
            args.rotation_mode, args.padding_mode)
        loss_1 = loss_1.item()
        
        loss_2 = 0
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask).item()
        
        loss_3 = smooth_loss(depth).item()

        loss_4 = 0
        if w4 > 0:
            loss_4, warped_sem, diff_sem = semantic_reconstruction_loss(
                target_sem, source_sem, intrinsics, depth, explainability_mask, 
                pose, args.rotation_mode, args.padding_mode)
            loss_4 = loss_4.item()
        
        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3 + w4 * loss_4
        
        if log_outputs and i in batches_to_log:  # log first output of wanted batches
            index = batches_to_log.index(i)
            if epoch == 0:
                for j, ref in enumerate(source):
                    tb_writer.add_image(
                        'val Input {}/{}'.format(j, index), tensor2array(target[0]), 0)
                    tb_writer.add_image(
                        'sem Input {}/{}'.format(j, index), tensor2array(target_sem[0]), 0)
                    tb_writer.add_image(
                        'val Input {}/{}'.format(j, index), tensor2array(ref[0]), 1)

            log_output_tensorboard_semantic(
                tb_writer, 'val', index, '', epoch, 1./disp, disp, warped[0], 
                diff[0], warped_sem[0], diff[0], explainability_mask)

        if log_outputs and i < len(val_loader)-1:
            step = args.batch_size*(args.sequence_length-1)
            poses[i * step:(i+1) * step] = pose.cpu().view(-1, 6).numpy()
            step = args.batch_size * 3
            disp_unraveled = disp.cpu().view(args.batch_size, -1)
            disp_values[i * step:(i+1) * step] = torch.cat(
                [disp_unraveled.min(-1)[0], disp_unraveled.median(-1)[0],
                disp_unraveled.max(-1)[0]]).numpy()

        losses.update([loss, loss_1, loss_4])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write(
                'valid: Time {} Loss {}'.format(batch_time, losses))
            
    if log_outputs:
        prefix = 'valid poses'
        coeffs_names = ['tx', 'ty', 'tz']
        if args.rotation_mode == 'euler':
            coeffs_names.extend(['rx', 'ry', 'rz'])
        elif args.rotation_mode == 'quat':
            coeffs_names.extend(['qx', 'qy', 'qz'])
        for i in range(poses.shape[1]):
            tb_writer.add_histogram('{} {}'.format(
                prefix, coeffs_names[i]), poses[:, i], epoch)
        tb_writer.add_histogram('disp_values', disp_values, epoch)
    logger.valid_bar.update(len(val_loader))
    
    return losses.avg, [
        'Validation Total loss', 'Validation Photo loss', 'Validation Sem loss']

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Structure from Motion Learner training on MP3D Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # I/O
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--print-freq', type=int,  default=10, metavar='N', 
        help='print frequency')
    parser.add_argument('--log-summary', default='log_summary.csv', metavar='PATH',
        help='csv where to save per-epoch train and valid stats')
    parser.add_argument('--log-full', default='log_full.csv', metavar='PATH',
        help='csv where to save per-gradient descent train stats')
    parser.add_argument('--log-output', action='store_true', 
        help='will log dispnet outputs and warped imgs at validation step')
    parser.add_argument('--training-output-freq', type=int, default=0, metavar='N',
        help='frequence for outputting dispnet outputs and warped imgs at training.'
        'if 0, will not output')
    
    # Hyper-parameters
    parser.add_argument('--sequence-length', type=int, default=3, metavar='N',
        help='sequence length for training')
    parser.add_argument('--epochs', type=int, default=115, metavar='N',
        help='number of total epochs to run')
    parser.add_argument('--epoch-size', type=int, default=0,  metavar='N',
        help='manual epoch size (will match dataset size if not set)')
    parser.add_argument('-b', '--batch-size', type=int, default=4, metavar='N',
        help='mini-batch size')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR', 
        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', type=float, default=0.999, metavar='M',
        help='beta parameters for adam')
    parser.add_argument('--weight-decay', type=float, default=0, metavar='W', 
        help='weight decay')
    parser.add_argument('--photo-loss-weight', type=float, default=1, metavar='W',
        help='weight for photometric loss')
    parser.add_argument('--mask-loss-weight', type=float, default=0.0, metavar='W',
        help='weight for explainabilty mask loss')
    parser.add_argument('--smooth-loss-weight', type=float, default=0.1, metavar='W',
        help='weight for disparity smoothness loss')
    parser.add_argument('--semantic-loss-weight', type=float, default=1.0, metavar='W',
        help='weight for disparity smoothness loss')
    parser.add_argument('--seed', type=int, default=0, 
        help='seed for random functions, and network initialization')
    
    parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], 
        default='euler', help='rotation mode for PoseExpnet : euler (yaw,pitch,roll)'
        ' or quaternion (last 3 coefficients)')
    parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
        help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
        ' zeros will null gradients outside target image.'
        ' border will only null gradients of the coordinate outside (x or y)')
    
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
        help='number of data loading workers')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
        help='evaluate model on validation set')
    parser.add_argument('--pin-memory', dest='pin_memory', action='store_true',
        help='evaluate model on validation set')
    parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, 
        metavar='PATH', help='path to pre-trained dispnet model')
    parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', 
        default=None, metavar='PATH', help='path to pre-trained Exp Pose net model')
    
    args = parser.parse_args()
    main(args)
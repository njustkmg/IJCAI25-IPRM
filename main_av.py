from utils.utils import setup_seed
from dataset.dataset import create_dataset
import copy
from torch.utils.data import DataLoader
from models.models import AVClassifier, Mixup_AVClassifier
from sklearn import metrics
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import re
import numpy as np
from tqdm import tqdm
import argparse
import os
import pickle
from operator import mod
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import time

lam_a_list, lam_v_list =[], []
lam_hat_a_list, lam_hat_v_list = [], []

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="CREMAD", type=str)
    parser.add_argument('--model', default='1+1+1', type=str)
    parser.add_argument('--n_classes', default=6, type=int, help = '6 for cremad, 31 for kineticSound')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--optimizer', default='sgd',
                        type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1,
                        type=float, help='decay coefficient')
    parser.add_argument('--ckpt_path', default='log_cd',
                        type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0,1,2,3',
                        type=str, help='GPU ids')
    parser.add_argument('--saved_model_name', type=str, required=True)
    parser.add_argument('--alpha', type=float, required=True)


    return parser.parse_args()



def compute_kl_divergence(logits1, logits2):
    
    probs1 = F.softmax(logits1, dim=-1)
    # probs2 = F.softmax(logits2, dim=-1)
    kl_divergence = torch.sum(probs1 * (F.log_softmax(logits1, dim=-1)-F.log_softmax(logits2, dim=-1)), dim=-1)
    
    return kl_divergence.mean()


def slerp(x, y, lambda_x, lambda_y):
    ############ geodesic mixup ###########
    x_norm = x / torch.norm(x, p=2, dim=1, keepdim=True)
    y_norm = y / torch.norm(y, p=2, dim=1, keepdim=True)
    
    dot = torch.sum(x_norm * y_norm, dim=1).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    theta = theta.unsqueeze(1).expand_as(x_norm)
    
    sin_theta = torch.sin(theta)
    sin_theta = sin_theta + 1e-8  # to avoid nan
    
    a = torch.sin(lambda_x * theta) / sin_theta
    b = torch.sin(lambda_y * theta) / sin_theta
    result = a * x_norm + b * y_norm
    return result

def cal_acc(args, prediction, label):
    n_classes = args.n_classes
    num = [0.0 for _ in range(n_classes)]
    acc = [0.0 for _ in range(n_classes)]

    for i, item in enumerate(label):
        ma = prediction[i].cpu().data.numpy()
        index_ma = np.argmax(ma)
        num[label[i]] += 1.0
        if index_ma == label[i]:
            acc[label[i]] += 1.0

    return sum(acc) / sum(num)

def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    print("Start training ... ")

    criterion = nn.CrossEntropyLoss()
    model.train()

    _loss = 0
    global lam_a_list, lam_v_list
    global lam_hat_a_list, lam_hat_v_list

    for step, (spec, images, label) in tqdm(enumerate(dataloader)):

        images = images.to(device)
        spec = spec.to(device)
        label = label.to(device)

        a, v = model(spec.unsqueeze(1).float(), images.float())

        out_a = model.head_audio(a)
        out_v = model.head_video(v)
        loss_a=criterion(out_a,label)
        loss_v=criterion(out_v,label)
        
        ############ Instantaneous Probing Phase ############
        with torch.no_grad():
            if len(lam_a_list) == 0:
                lam_a = 0.5
                lam_v = 0.5
            else:
                lam_a = lam_a_list[-1] * args.alpha + lam_hat_a_list[-1] * (1-args.alpha)
                lam_v = lam_v_list[-1] * args.alpha + lam_hat_v_list[-1] * (1-args.alpha)
            
            mixup_f = slerp(a, v, lam_a, lam_v)
            out_tmp = model.head_mixup(mixup_f)

            KL_a = compute_kl_divergence(out_a, out_tmp)
            KL_v = compute_kl_divergence(out_v, out_tmp)

            lam_hat_a = KL_v/(KL_a+KL_v)
            lam_hat_v = KL_a/(KL_a+KL_v)

            lam_a_list.append(lam_a)
            lam_v_list.append(lam_v)
            lam_hat_a_list.append(lam_hat_a.cpu())
            lam_hat_v_list.append(lam_hat_v.cpu())
        ####################################################

        ############ Rebalanced Learning Phase #############
        mixup_f = slerp(a, v, lam_hat_a, lam_hat_v)
        out_mm = model.head_mixup(mixup_f)

        loss_mm=criterion(out_mm,label)

        loss=loss_mm+loss_a+loss_v
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        ####################################################
        _loss += loss.item()

    scheduler.step()
    return _loss / len(dataloader)

def valid(args, model, device, dataloader):
    print("Start validation ...")

    n_classes = args.n_classes

    global lam_a_list, lam_v_list
    global lam_hat_a_list, lam_hat_v_list

    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a= [0.0 for _ in range(n_classes)]
        acc_v= [0.0 for _ in range(n_classes)]

        for step, (spec, images, label) in tqdm(enumerate(dataloader)):

            spec = spec.to(device)
            images = images.to(device)
            label = label.to(device)

            a, v = model(spec.unsqueeze(1).float(), images.float())

            mixup_f = slerp(a, v, lam_a_list[-1], lam_v_list[-1])

            out_a = model.head_audio(a)
            out_v = model.head_video(v)

            out_mm = model.head_mixup(mixup_f)

            prediction= out_a + out_v + out_mm
            prediction_audio=out_a
            prediction_visual=out_v

            for i, item in enumerate(label):

                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0
                
                ma_audio=prediction_audio[i].cpu().data.numpy()
                index_ma_audio = np.argmax(ma_audio)
                if index_ma_audio == label[i]:
                    acc_a[label[i]] += 1.0

                ma_visual=prediction_visual[i].cpu().data.numpy()
                index_ma_visual = np.argmax(ma_visual)
                if index_ma_visual == label[i]:
                    acc_v[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    global lam_a_list, lam_v_list
    global lam_hat_a_list, lam_hat_v_list

    args = get_arguments()
    print(args)
    setup_seed(args.random_seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')
    
    if args.dataset == 'CREMAD':
        train_dataset, test_dataset = create_dataset('CREMAD')
        args.n_classes = 6

    model = Mixup_AVClassifier(args)
    model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=16, pin_memory=False)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=16)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    print(len(train_dataloader))

    best_acc = 0

    if args.train:

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            batch_loss = train_epoch(
                args, epoch, model, device, train_dataloader, optimizer, scheduler, None)

            print("Loss: {:.4f}".format(batch_loss))
            
            acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

            print("Test Acc: {:.4f}, Test Acc a: {:.4f}, Test Acc v: {:.4f}".format(
                    acc, acc_a, acc_v))     
           
            if acc > best_acc:
                best_acc = acc
                model_name = '{}_of_{}_Best_batch{}_lr{}.pth'.format(
                    args.saved_model_name, args.dataset, args.batch_size, args.learning_rate)
                saved_dict = {'saved_epoch': epoch,
                                'acc': acc,
                                'lam_a': lam_a_list[-1],
                                'lam_v': lam_v_list[-1],
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict()}
                save_dir = os.path.join(args.ckpt_path, model_name)
                torch.save(saved_dict, save_dir)
                print("Model saved as {}".format(save_dir))

if __name__ == "__main__":
    main()

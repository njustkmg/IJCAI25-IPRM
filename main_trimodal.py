import argparse
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataset.dataset import create_dataset
from models.RODModel import MixupClassfier
from utils.utils import setup_seed, weight_init
from tqdm import tqdm
import torch.nn.functional as F
import random
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score, accuracy_score


lam_list = [[[] for _ in range(3)]for _ in range(3)]

lam_hat_list= [[[] for _ in range(3)]for _ in range(3)]

lam = [[torch.tensor(0.5) for _ in range(3)]for _ in range(3)]


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="NVGesture", type=str)
    parser.add_argument('--model', default='1+1+1', type=str)
    parser.add_argument('--n_classes', default=6, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--optimizer', default='sgd',
                        type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1,
                        type=float, help='decay coefficient')
    parser.add_argument('--ckpt_path', default='log_cd/nvGesture',
                        type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0,1,2,3',
                        type=str, help='GPU ids')
    parser.add_argument('--saved_model_name', type=str, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--mixup_method', type=str, help='single_cls, tri_cls', required=True)

    return parser.parse_args()


def compute_kl_divergence(logits1, logits2):
    
    probs1 = F.softmax(logits1, dim=-1)
    # probs2 = F.softmax(logits2, dim=-1)
    
    kl_divergence = torch.sum(probs1 * (F.log_softmax(logits1, dim=-1) - F.log_softmax(logits2, dim=-1)), dim=-1)
    
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


def train_epoch(args, epoch, model, device, train_dataloader, optimizer, scheduler, writer=None):    
    print('Start Training')

    global lam_list, lam_hat_list, lam
    criterion = nn.CrossEntropyLoss()
    _total_loss = 0

    model.train()
    
    for step, bag in tqdm(enumerate(train_dataloader)):
        rgb = bag[0].float().to(device)
        of = bag[1].to(device)
        depth = bag[2].to(device)
        label = bag[3].to(device)

        m1, m2, m3= model(rgb, of, depth)
        out_m1 = model.head_m1(m1)
        out_m2 = model.head_m2(m2)
        out_m3 = model.head_m2(m3)
        loss1 = criterion(out_m1, label)
        loss2 = criterion(out_m2, label)
        loss3 = criterion(out_m3, label)

        uni_feature= [m1,m2,m3]
        uni_out = [out_m1, out_m2, out_m3]

        s_i, s_j = 0, 0 # for single_cls strategy

        ############# probe phase #############
        with torch.no_grad():
            if args.mixup_method == 'tri_cls':
                for i in range(0,3):
                    for j in range(i+1,3):
                        # if first iteration, set 0.5,0.5
                        if len(lam_list[i][j]) == 0:
                            lam[i][j] = torch.tensor(0.5)
                            lam[j][i] = torch.tensor(0.5)
                        # else find last k1,k2, and perform ema
                        else:
                            lam[i][j] = lam_list[i][j][-1] * args.alpha + lam_hat_list[i][j][-1] * (1-args.alpha)
                            lam[j][i] = lam_list[j][i][-1] * args.alpha + lam_hat_list[j][i][-1] * (1-args.alpha)

                        mixup_f = slerp(uni_feature[i],uni_feature[j],lam[i][j],lam[j][i])

                        if i+j==1:
                            out_tmp = model.head_mixup_1(mixup_f)
                        elif i+j==2:
                            out_tmp = model.head_mixup_2(mixup_f)
                        else: # i+j==3
                            out_tmp = model.head_mixup_3(mixup_f)   

                        KL_m1 = compute_kl_divergence(uni_out[i], out_tmp)
                        KL_m2 = compute_kl_divergence(uni_out[j], out_tmp)
                        lam_hat_m1 = KL_m2/(KL_m1+KL_m2)
                        lam_hat_m2 = KL_m1/(KL_m1+KL_m2)

                        lam_list[i][j].append(lam[i][j])
                        lam_list[j][i].append(lam[j][i])

                        lam_hat_list[i][j].append(lam_hat_m1.cpu())
                        lam_hat_list[j][i].append(lam_hat_m2.cpu())

            elif args.mixup_method == 'single_cls':
                nums = [0,1,2]
                i, j = random.sample(nums, 2)
                s_i, s_j = i, j

                # if first iteration, set 0.5,0.5
                if len(lam_list[i][j]) == 0:
                    lam[i][j] = torch.tensor(0.5)
                    lam[j][i] = torch.tensor(0.5)

                else:
                    lam[i][j] = lam[i][j] * args.alpha + lam_hat_list[i][j][-1] * (1-args.alpha)
                    lam[j][i] = lam[j][i] * args.alpha + lam_hat_list[j][i][-1] * (1-args.alpha)
                
                mixup_f = slerp(uni_feature[i],uni_feature[j],lam[i][j],lam[j][i])
                out_tmp = model.head_mixup(mixup_f)
                KL_m1 = compute_kl_divergence(uni_out[i], out_tmp)
                KL_m2 = compute_kl_divergence(uni_out[j], out_tmp)
                lam_hat_m1 = KL_m2/(KL_m1+KL_m2)
                lam_hat_m2 = KL_m1/(KL_m1+KL_m2)

                lam_list[i][j].append(lam[i][j])
                lam_list[j][i].append(lam[j][i])

                lam_hat_list[i][j].append(lam_hat_m1.cpu())
                lam_hat_list[j][i].append(lam_hat_m2.cpu())
        #########################################

        ############# reblance phase #############      
        if args.mixup_method=='tri_cls':
            all_mixup_f = []
            for i in range(0,3):
                for j in range(i+1,3):
                    all_mixup_f.append(slerp(uni_feature[i], uni_feature[j], lam_hat_list[i][j][-1], lam_hat_list[j][i][-1]))

            out_mm1 = model.head_mixup_1(all_mixup_f[0])
            out_mm2 = model.head_mixup_2(all_mixup_f[1])
            out_mm3 = model.head_mixup_3(all_mixup_f[2])
            loss_mm = criterion(out_mm1, label) + criterion(out_mm2, label) + criterion(out_mm3, label)
      
        elif args.mixup_method=='single_cls':
            mixup_f = slerp(uni_feature[s_i], uni_feature[s_j], lam_hat_list[s_i][s_j][-1], lam_hat_list[s_j][s_i][-1])
            out_mm = model.head_mixup(mixup_f)
            loss_mm = criterion(out_mm, label)

        loss = loss1+loss2+loss3+loss_mm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #########################################
        _total_loss += loss.item()

    scheduler.step()
    return _total_loss / len(train_dataloader)

def valid(args, model, device, test_dataloader):

    print("Start validation ...")

    global lam_list, lam_hat_list, lam 
    all_preds = []
    all_preds_m1 = []
    all_preds_m2 = []
    all_preds_m3 = []

    all_labels = []
    
    model.eval()
    for step, bag in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            rgb = bag[0].float().to(device)
            of = bag[1].to(device)
            depth = bag[2].to(device)
            label = bag[3].to(device)

            m1, m2, m3= model(rgb, of, depth)
            out_m1 = model.head_m1(m1)
            out_m2 = model.head_m2(m2)
            out_m3 = model.head_m2(m3)

            uni_feature= [m1,m2,m3]

            all_mixup_f = []
            for i in range(0,3):
                for j in range(i+1,3):
                    all_mixup_f.append(slerp(uni_feature[i], uni_feature[j], lam_list[i][j][-1], lam_list[j][i][-1]))

            if args.mixup_method == 'tri_cls':
                out_mm1 = model.head_mixup_1(all_mixup_f[0])
                out_mm2 = model.head_mixup_2(all_mixup_f[1])
                out_mm3 = model.head_mixup_3(all_mixup_f[2])
                out_mm = (out_mm1+out_mm2+out_mm3)/3
                # out_mm = out_m1+out_m2+out_m3
            elif args.mixup_method =='single_cls':
                out_mm1 = model.head_mixup(all_mixup_f[0])
                out_mm2 = model.head_mixup(all_mixup_f[1])
                out_mm3 = model.head_mixup(all_mixup_f[2])
                out_mm = (out_mm1+out_mm2+out_mm3)/3

            out = out_m1 + out_m2 + out_m3 + out_mm
            preds    = torch.max(out, dim=1)[1]
            preds_m1 = torch.max(out_m1, dim=1)[1]
            preds_m2 = torch.max(out_m2, dim=1)[1]
            preds_m3 = torch.max(out_m3, dim=1)[1] 

            all_preds.extend(preds.tolist())
            all_preds_m1.extend(preds_m1.tolist())
            all_preds_m2.extend(preds_m2.tolist())
            all_preds_m3.extend(preds_m3.tolist())

            all_labels.extend(label.tolist())

    all_preds_np = np.array(all_preds)
    all_preds_m1_np = np.array(all_preds_m1)
    all_preds_m2_np = np.array(all_preds_m2)
    all_preds_m3_np = np.array(all_preds_m3)
    all_labels_np = np.array(all_labels)

    accuracy    = accuracy_score(all_labels_np, all_preds_np)
    accuracy_m1 = accuracy_score(all_labels_np, all_preds_m1_np)
    accuracy_m2 = accuracy_score(all_labels_np, all_preds_m2_np)    
    accuracy_m3 = accuracy_score(all_labels_np, all_preds_m3_np)

    return accuracy, accuracy_m1, accuracy_m2, accuracy_m3



def main():

    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')
    
    train_dataset, test_dataset = create_dataset('NVGesture')
    args.n_classes = 25

    model = MixupClassfier(args)
    # model.apply(weight_init)
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

            print('Train Loss: {:.4f}'.format(batch_loss))
            
            acc, acc_m1, acc_m2, acc_m3 = valid(args, model, device, test_dataloader)

            print('Test Acc:{:.4f}, Test rgb Acc:{:.4f}, Test of Acc:{:.4f}, Test depth Acc:{:.4f}'.format(acc, acc_m1, acc_m2, acc_m3))
            
            if acc > best_acc:
                best_acc = acc
                model_name = '{}_of_{}_Best_batch{}_lr{}.pth'.format(
                    args.saved_model_name, args.dataset, args.batch_size, args.learning_rate)
                saved_dict = {'saved_epoch': epoch,
                                'acc': acc,
                                'lam': lam,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict()}
                save_dir = os.path.join(args.ckpt_path, model_name)
                torch.save(saved_dict, save_dir)
                print('Model saved at', save_dir)

if __name__ == "__main__":
    main()
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import collections
from collections import OrderedDict
from glob import glob
import copy
import natsort
import cv2

import albumentations as A
from torchvision import transforms
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
from pytorch_metric_learning import losses
from pytorch_lightning import seed_everything

from networks.unet2d import Unet2D, UNet_OG
'''Efficient-Unet'''
from networks.EfficientUnet.efficientunet import *
from utils.losses import *
from utils.util import _eval_dice, _eval_dice_mri, _eval_haus, _connectivity_region_analysis, parse_fn_haus
from utils.metrics import dice_coef_metric, iou_metric
from utils.aggregators import *
from utils.gam_select import *
from utils.mislabel import *
from dataloaders.lgg_dataloader import Dataset_LGG, normalize
from dataloaders.transforms import train_tfm, eval_tfm
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,  default='xxxx', help='model_name')
parser.add_argument('--max_epoch', type=int,  default=150, help='maximum epoch number to train')
parser.add_argument('--local_epoch', type=int,  default=2, help='local epoch number')
parser.add_argument('--client_num', type=int, default=4, help='client number per gpu')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu') #16
parser.add_argument('--model_name', type=str,  default='unet', help='type of model')
parser.add_argument('--clip_value', type=float,  default=1, help='clip value') #10
parser.add_argument('--local_step_size', type=float,  default=5e-4, help='local step size') #1e-3
parser.add_argument('--base_lr', type=float,  default=5e-4, help='base learning rate') #1e-3, 1e-4
parser.add_argument('--loss_fn', type=str,  default='bce_dice_loss', help='loss function')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')#1337
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--display_freq', type=int, default=5, help='display frequency')
parser.add_argument('--aggr', type=str,  default='mean', help='aggregator')
parser.add_argument('--gamma', type=float,  default=1e-4, help='value of gamma')
parser.add_argument('--mode', type=str,  default='standard', help='standard or fedbn')
parser.add_argument('--lambd', type=float,  default=1, help='lambda value')
parser.add_argument('--mislabel_rate', type=float, default=0, help='mislabel ratio, default to 0')
parser.add_argument('--num_workers', type=int, default=4, help='number of threads') #2, 4
parser.add_argument('--state', type=str, default='train', help='train or test')

args = parser.parse_args()

snapshot_path = "./output/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
expr = args.exp
batch_size = args.batch_size * len(args.gpu.split(','))
local_step_size = args.local_step_size
clip_value = args.clip_value
model_name = args.model_name
loss_fn = args.loss_fn
base_lr = args.base_lr
client_num = args.client_num
local_epoch = args.local_epoch
max_epoch = args.max_epoch
display_freq = args.display_freq
n_workers = args.num_workers
lambd = args.lambd
mislabel_rate = args.mislabel_rate

client_name = [f'client{cl+1}' for cl in range(client_num)]
clients_lr = [local_step_size]*len(client_name)
lambd_list = [lambd, lambd, lambd, lambd] #10, 0, 0.1
gamma_list = [args.gamma]*client_num #initial gamma list

#number of slices
slice_num = np.array([1878, 1029, 640, 358])
for c, name in enumerate(client_name):
    print(name,':', slice_num[c])

#data size
volume_size = [256, 256, 3]
#client index
source_site_idx = list(range(client_num))
#number of slices samples client weights
client_weight = slice_num[source_site_idx] / np.sum(slice_num[source_site_idx])
client_weight = np.round(client_weight, decimals=2)
print(client_weight)
num_classes = 1

#set seed
seed_everything(args.seed)


def update_global_model(net_server, net_clients, diff, epc, prev, client_weight=None, aggr='mean', mode='standard'):
    print(f'Aggregators:{aggr}, Mode:{mode}')
    '''
    mean aggregator
    '''
    if aggr == 'mean':
        if mode == 'fedbn':
            client_weight = torch.from_numpy(client_weight).to(device)
            for lyr, (key, _) in enumerate(net_server.named_parameters()):
                #do not aggregate bn layer
                if 'bn' not in key:
                    agg_grad = Variable(torch.Tensor(np.zeros(net_server.state_dict()[key].data.shape)), requires_grad=False).to(device)
                    #compute aggr change of parameters
                    for i in range(client_num):
                        grads = diff[i][lyr]*client_weight[i]
                        agg_grad.data.add_(grads)
                    #update global model
                    net_server.state_dict()[key].data.add_(agg_grad)
                    updates = net_server.state_dict()[key].data
                    #send glob to local
                    for i in range(client_num):
                        net_clients[i].state_dict()[key].data.copy_(updates)
                        
        if mode == 'standard':
            client_weight = torch.from_numpy(client_weight).to(device)
            for lyr, (key, _) in enumerate(net_server.named_parameters()):
                agg_grad = Variable(torch.Tensor(np.zeros(net_server.state_dict()[key].data.shape)), requires_grad=False).to(device)
                #compute aggr change of parameters
                for i in range(client_num):
                    grads = diff[i][lyr]*client_weight[i]
                    agg_grad.data.add_(grads)
                #update global model
                net_server.state_dict()[key].data.add_(agg_grad)
                updates = net_server.state_dict()[key].data
                #send glob to local
                for i in range(client_num):
                    net_clients[i].state_dict()[key].data.copy_(updates)
    '''
    simple gamma mean aggregator
    '''
    if aggr == 'gam_mean':
        if mode == 'fedbn':
            gam_mean_full = []
            cnt = 0
            for lyr, (key, _) in enumerate(net_server.named_parameters()):
                if 'bn' not in key:
                    points = torch.vstack([diff[i][lyr].view(-1).unsqueeze(0) for i in range(client_num)]).data.cpu().numpy()
                    '''gamma selection'''
                    if epc == 0:
                        gamma = args.gamma
                    else:
                        #print(points.shape, prev[cnt].shape)
                        gamma = gam_select(points, prev[cnt])
                        #print('gamma:', gamma)
                    
                    '''gamma mean'''
                    gam_mean = gamma_mean(points, weights=None, history_points=None, compute='simple', gamma=gamma, max_iter=5, 
                                                     tol=1e-7, remove=False, beta=0.1, initial='median') #gamma=0.001, 0.01, 0.1, 1
                   
                    gam_mean_full.append(gam_mean)
                    agg_grad = torch.Tensor(gam_mean).view(net_server.state_dict()[key].shape).to(device)
                    
                    #update global model
                    net_server.state_dict()[key].data.add_(agg_grad)
                    updates = net_server.state_dict()[key].data
                    #send glob to local
                    for i in range(client_num):
                        net_clients[i].state_dict()[key].data.copy_(updates)
                    cnt += 1
                    
            return gam_mean_full
        
        if mode == 'standard':
            gam_mean_full = []
            for lyr, (key, _) in enumerate(net_server.named_parameters()):
                points = torch.vstack([diff[i][lyr].view(-1).unsqueeze(0) for i in range(client_num)]).data.cpu().numpy()
                '''gamma selection'''
                if epc == 0:
                    gamma = args.gamma
                else:
                    gamma = gam_select(points, prev[cnt])
                gam_mean = gamma_mean(points, weights=None, history_points=None, compute='simple', gamma=gamma, max_iter=5, 
                           tol=1e-7, remove=False, beta=0.1, initial='median') #gamma=0.001, 0.01, 0.1, 1
                gam_mean_full.append(gam_mean)
                agg_grad = torch.Tensor(gam_mean).view(net_server.state_dict()[key].shape).to(device)

                #update global model
                net_server.state_dict()[key].data.add_(agg_grad)
                updates = net_server.state_dict()[key].data
                #send glob to local
                for i in range(client_num):
                    net_clients[i].state_dict()[key].data.copy_(updates)
            return gam_mean_full
    '''
    geometric median
    '''
    if aggr == 'geomed':
        if mode == 'fedbn':
            for lyr, (key, _) in enumerate(net_server.named_parameters()):
                if 'bn' not in key:
                    points = torch.vstack([diff[i][lyr].view(-1).unsqueeze(0) for i in range(client_num)]).data.cpu().numpy()
                    gam_mean = geometric_median(points)
                    agg_grad = torch.Tensor(gam_mean).view(net_server.state_dict()[key].shape).to(device)
                    
                    #update global model
                    net_server.state_dict()[key].data.add_(agg_grad)
                    updates = net_server.state_dict()[key].data
                    #send glob to local
                    for i in range(client_num):
                        net_clients[i].state_dict()[key].data.copy_(updates)
        
        if mode == 'standard':
            for lyr, (key, _) in enumerate(net_server.named_parameters()):
                points = torch.vstack([diff[i][lyr].view(-1).unsqueeze(0) for i in range(client_num)]).data.cpu().numpy()
                gam_mean = geometric_median(points)
                agg_grad = torch.Tensor(gam_mean).view(net_server.state_dict()[key].shape).to(device)

                #update global model
                net_server.state_dict()[key].data.add_(agg_grad)
                updates = net_server.state_dict()[key].data
                #send glob to local
                for i in range(client_num):
                    net_clients[i].state_dict()[key].data.copy_(updates)


def evaluation(loader, test_net, run='test'):
    test_net.eval()
    eval_pred_out, eval_mask_out, val_losses = [], [], []
    
    with torch.no_grad():
        for step, batch in enumerate(loader):
            image_batch = batch['image'].to(device)
            mask_batch = batch['label'].to(device)
            '''mislabeling'''
            if (run == 'val') and (mislabel_rate > 0):
                mis_idx = round(mislabel_rate*mask_batch.shape[0]) #compute mislabel idx for normal/mislabel split
                mask_batch[:mis_idx] = mislabeling(mask_batch[:mis_idx], step).to(device)
            
            logit = test_net(image_batch)
            '''check loss function'''
            if loss_fn == 'auto_gamma_dice_loss':
                loss, _ = criterion(logit, mask_batch)
            else:
                loss = criterion(logit, mask_batch)
                
            #loss = criterion_local(logit, mask)
            pred_y = logit.cpu().detach().numpy() #(1, 1, 256, 256)
            #prediction with threshold
            pred_y[pred_y>0.3] = 1
            pred_y[pred_y<0.3] = 0
            eval_pred_out.append(pred_y)
            eval_mask_out.append(mask_batch.cpu().numpy())
            val_losses.append(loss.cpu().numpy())
    
    #stack full predicted masks
    eval_outs = np.vstack(eval_pred_out)
    eval_masks = np.vstack(eval_mask_out)
    
    eval_mean_dice = dice_coef_metric(eval_outs, eval_masks)
    eval_mean_iou = iou_metric(eval_outs, eval_masks)
    eval_mean_loss = np.mean(val_losses)
    
    return eval_mean_dice, eval_mean_iou, eval_mean_loss

if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    #set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(snapshot_path + '/model'):
        os.makedirs(snapshot_path + '/model')
    
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    # define dataset, model, optimizer for each client
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    
    '''
    set server model
    '''
    if model_name == 'unet':
        net_server = UNet_OG(3,1).to(device)
    if model_name == 'effunet':
        net_server = efficientunet.get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=True).to(device)
    
    
    '''
    set clients: add dataloader, model, optimizer
    '''
    train_loader_clients, val_loader_clients, test_loader_clients = [], [], []
    net_clients = []
    net_person = []
    optimizer_clients = []
    scheduler_clients = []
    
    for client_idx in range(client_num):
        #LGG
        if client_idx != 4:
            image_list = glob('./dataset/LGG/{}/data_npy/*'.format(client_name[client_idx]))
            #train test split
            train, test = train_test_split(image_list, test_size=0.1, random_state=args.seed)
            train, val = train_test_split(train, test_size=0.1, random_state=args.seed)
            #we can perform augmentation
            train_set = Dataset_LGG(train, train_tfm)
            val_set = Dataset_LGG(val, eval_tfm)
            test_set = Dataset_LGG(test, eval_tfm)
            
            #dataloader
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=n_workers, pin_memory=True, worker_init_fn=worker_init_fn)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,  num_workers=n_workers, pin_memory=True, worker_init_fn=worker_init_fn)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,  num_workers=n_workers, pin_memory=True, worker_init_fn=worker_init_fn)
            
        #net = Unet2D(num_classes=1).to(device)
        if model_name == 'unet':
            net  = UNet_OG(3,1).to(device)
        if model_name == 'effunet':
            net = efficientunet.get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=True).to(device)
            
        optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
        train_loader_clients.append(train_loader)
        val_loader_clients.append(val_loader)
        test_loader_clients.append(test_loader)
        net_clients.append(net)
        net_person.append(net)
        optimizer_clients.append(optimizer)
        scheduler_clients.append(scheduler)
        
    
    '''
    loss function
    '''
    if loss_fn == 'bce_dice_loss':
        criterion = bce_dice_loss
    if loss_fn == 'focal_loss':
        criterion = focal_loss
    if loss_fn == 'gamma_dice_loss':
        criterion = gamma_dice_loss
    if loss_fn == 'auto_gamma_dice_loss':
        criterion = auto_gamma_dice_loss
    
    '''
    start federated learning
    '''
    if args.state == 'train':
        print('----Start FL Training----')
        print(f'loss_function: {loss_fn}')
        writer = SummaryWriter(snapshot_path+'/log')
        lr_ = base_lr
        best_local_loss = [1e3]*len(client_name)
        
        for epoch_num in tqdm(range(max_epoch), ncols=70):
            print('\n', '-----'*10)
            '''stored change of parameters'''
            diff_params = []
            
            for client_idx in source_site_idx:
                dataloader_current = train_loader_clients[client_idx]
                net_current = net_clients[client_idx]
                net_local = net_person[client_idx]
                #store prev net and params
                prev_net = copy.deepcopy(net_current)
                
                net_current.train()
                optimizer_current = optimizer_clients[client_idx]

                time1 = time.time()
                iter_num = 0
                '''store all batches updated gamma'''
                gam_update_full = [] 
                #local epochs
                '''1. update local global model'''
                for lc_epc in range(local_epoch): 
                    for i_batch, sampled_batch in enumerate(dataloader_current):
                        time2 = time.time()

                        # obtain training data
                        #volume_batch: (16, 3, 256,256), label_batch: (16, 1, 256, 256)
                        volume_batch_raw, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
                        '''if mislabeling'''
                        if mislabel_rate > 0:
                            mis_idx = round(mislabel_rate*label_batch.shape[0]) #compute mislabel idx for normal/mislabel split
                            sd = epoch_num + i_batch #seed for mislabeling
                            label_batch[:mis_idx] = mislabeling(label_batch[:mis_idx], sd).to(device)
                        
                        # obtain updated parameter at inner loop and extract second last layer embedding: Z
                        '''
                        with original images 
                        '''
                        outputs = net_current(volume_batch_raw)
                        #dice loss: with old weights         
                        if loss_fn == 'auto_gamma_dice_loss':
                            global_loss, gam_update = criterion(outputs, label_batch, gamma_list[client_idx])
                            
                            #append last local epoch for updating local global model
                            if lc_epc == (local_epoch-1):
                                gam_update_full.append(gam_update)
                    
                        else:
                            #print(outputs.shape, label_batch.shape)
                            global_loss = criterion(outputs, label_batch)
                        '''
                        local param weights update
                        '''
                        #with torch.autograd.detect_anomaly():
                        optimizer_current.zero_grad()
                        global_loss.backward() #retain_graph=True
                        #gradient clipping
                        torch.nn.utils.clip_grad_norm_(net_current.parameters(), max_norm=1) #10
                        optimizer_current.step()
                
                '''update auto gamma selection'''
                if loss_fn == 'auto_gamma_dice_loss':
                    gamma_list[client_idx] = np.mean(gam_update_full)
                    #print(f'gamma for client {client_idx+1}:', gamma_list[client_idx])
                    
                #cosine annealing scheduler
                scheduler_clients[client_idx].step()

                #compute change of parameters of each layer
                temp = []
                for param in zip(prev_net.parameters(), net_current.parameters()):
                    changes = Variable(torch.Tensor(np.zeros(param[0].shape)), requires_grad=False).to(device)
                    diff = param[1] - param[0]
                    changes.data.add_(diff.data)
                    temp.append(changes)
                diff_params.append(temp)

                '''2. update local'''
                for _ in range(local_epoch):

                    outs = net_local(volume_batch_raw)
                    if loss_fn == 'auto_gamma_dice_loss':
                        local_loss, _ = criterion(outputs, label_batch, gamma_list[client_idx])
                    else:
                        local_loss = criterion(outputs, label_batch)
                    
                    #local_loss.backward()

                    #local_grads = torch.autograd.grad(local_loss, net_local.parameters(), retain_graph=True)
                    '''update local by layers'''
                    #for (server_param, local_param, grad) in zip(net_server.parameters(), net_local.parameters(), local_grads):
                    for (server_dict, client_dict) in zip(net_server.named_parameters(), net_local.named_parameters()):
                        '''get server, client params '''
                        key, server_param = server_dict[0], server_dict[1]
                        client_param = client_dict[1]
                        '''compute gradient with regularization term and update params'''
                        eff_grad = client_param.grad + lambd_list[client_idx]*(client_param-server_param)
                        post_param = client_param - torch.mul(clients_lr[client_idx], eff_grad)
                        '''update to net local'''
                        net_local.state_dict()[key].data.copy_(post_param)
                        
                #learning rate decay
                clients_lr[client_idx] = clients_lr[client_idx] / np.sqrt(1 + clients_lr[client_idx] * 0.9 * epoch_num)
                #clients_lr[client_idx] = 0.99*clients_lr[client_idx]
                
                #update local model
                net_person[client_idx] = net_local
                #update to net_clients list
                net_clients[client_idx] = net_current
                
                ## local evaluation
                with open(os.path.join(snapshot_path, f'eval_result_client{client_idx}.txt'), 'a') as f:
                    #dataloader
                    eval_loader = val_loader_clients[client_idx]
                    #dice, iou, val_loss = evaluation(eval_loader, net_clients[client_idx])
                    dice, iou, val_loss = evaluation(eval_loader, net_person[client_idx], run='val')
                    dice, iou, val_loss = float(dice), float(iou), float(val_loss)

                    print("epoch {}, client{} validation: dice:{}, iou:{}, val_loss:{}".format(epoch_num, client_idx+1, round(dice, 3), round(iou,3), round(val_loss,3)))
                    #write to file
                    print("epoch {}, client{} validation: dice:{}, iou:{}, val_loss:{}".format(epoch_num, client_idx+1, round(dice, 3), round(iou,3), round(val_loss,3)), file=f)
                ## save model
                save_mode_path = os.path.join(snapshot_path + '/model', f'client_{client_idx}_epoch_{epoch_num}' + '.pth')
                #save client model 
                #torch.save(net_clients[client_idx].state_dict(), save_mode_path)
                torch.save(net_person[client_idx].state_dict(), save_mode_path)
                #logging.info("save model to {}".format(save_mode_path))


            ## model aggregation
            if epoch_num == 0:
                prev = update_global_model(net_server, net_clients, diff_params, epoch_num, prev=None, client_weight=client_weight, aggr=args.aggr, mode=args.mode)
            else:
                prev = update_global_model(net_server, net_clients, diff_params, epoch_num, prev_cp, client_weight=client_weight, aggr=args.aggr, mode=args.mode)
            prev_cp = copy.deepcopy(prev)

        print('-----Finish FL------')
        writer.close()
    
    '''
    Testing
    '''
    if args.state =='test':
        print('-----Testing------')
        for epc in range(9, max_epoch, 5):
        #for epc in range(115, 116, 1):
            print(f'Ckpt EPOCH: {epc}')
            for c in range(client_num):
                ckpt_path = f'client_{c}_epoch_{epc}.pth'
                '''model type'''
                if model_name == 'unet':
                    net  = UNet_OG(3,1).to(device)
                if model_name == 'effunet':
                    net = efficientunet.get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=True).to(device)
                    
                net.load_state_dict(torch.load(os.path.join(f'./output/{expr}/model/', ckpt_path)))
                #dataloader
                testing_loader = test_loader_clients[c]
                dice, iou, test_loss = evaluation(testing_loader, net, run='test')
                dice, iou, test_loss = float(dice), float(iou), float(test_loss)

                print("Testing: client{} testing: dice:{}, iou:{}, val_loss:{}".format(c+1, round(dice, 3), round(iou,3), round(test_loss,3)))

    

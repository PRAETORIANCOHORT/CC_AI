from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
from torch.autograd import Variable
from torchvision import models, transforms
import torch.nn.functional as F
import pandas as pd
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import random
from PIL import Image 
from sklearn import metrics
import glob
import os
import time
import json
import apex
from apex import amp
from apex.fp16_utils import *
from apex.parallel import DistributedDataParallel
from apex.multi_tensor_apply import multi_tensor_applier
from torch.utils.data import Dataset, DataLoader as DL, Sampler
from torch.utils.data.distributed import DistributedSampler
import torchvision
import random
import copy
import shutil
from thop import profile
import sys
import cv2
from torchvision.utils import save_image
from torch.autograd import Variable
from scipy import stats
import datetime
import math


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of multiple Instance learning')
    parser.add_argument('--path', default='None', type=str)
    parser.add_argument('--epoch', default='None', type=str)
    parser.add_argument('--sample', default='None', type=str)
    parser.add_argument('--fold', default='None', type=str)
    parser.add_argument('--mag', default='5', type=str)
    parser.add_argument('--model', default=18, type=str)
    parser.add_argument('--test_limit', default=50, type=int)
    parser.add_argument('--extd', default=50, type=int)
    parser.add_argument('--save_path', default='../CAM_CC', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init_method', type=str)
    
    return parser.parse_args()


args = get_parser()
epoch = args.epoch
with open('../script/pat_labels.json3') as f:
    data_map = json.load(f)


data_path = args.path
pat_slide_all = pd.read_excel('../script/pat_slide_all.xlsx')
avlb_slide = os.listdir(data_path)

avlb_slide_df = pat_slide_all[pat_slide_all['slide'].isin(avlb_slide)]
all_id_list = list(set(avlb_slide_df['pat'].tolist()))

random.seed(int(args.sample.split('_')[1]))
random.shuffle(all_id_list)

F0_valid_pat = all_id_list[int(0.8*len(all_id_list)):]
test0_df = avlb_slide_df[avlb_slide_df['pat'].isin(F0_valid_pat)]
test0_label = list(set(test0_df['slide'].tolist()))

old_neg = list(set(pd.read_excel('../script/old_slide_all.xlsx')['old_slide'].tolist())&set(test0_label))
test0_label = list(set(test0_label)-set(old_neg))
val_label = list(set(test0_label)-set([f.split('_')[3] for f in os.listdir(f'../script/df_final/X{args.mag}/')]))

cluster_2 = [f.split('|')[1] for f in os.listdir('../cluster/patches-8-encoder_2349/1')]
cluster_4 = [f.split('|')[1] for f in os.listdir('../cluster/patches-8-encoder_2349/3')]
cluster_6 = [f.split('|')[1] for f in os.listdir('../cluster/patches-8-encoder_2349/5')]

# val_label = list(set(val_label)&(set(cluster_2)|set(cluster_4)|set(cluster_6)))
val_label = list((set(cluster_2)|set(cluster_4)|set(cluster_6))-set(os.listdir('/gputemp/ToWZP/CAM_CC/CAM_4')))
val_label.sort()

print('Number of slides:', len(val_label))
print(val_label)


test_transform = transforms.Compose([
#             transforms.CenterCrop(384),
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize([0.6522, 0.3254, 0.6157], [0.2044, 0.2466, 0.1815])
        ])

img_transform = transforms.Compose([
#             transforms.CenterCrop(384),
            transforms.Resize(299)
        ])


def get_loc(img_name):
    return np.array(list(map(int,img_name.split('/')[-1].split('.')[0].split('_'))))


def resampling(list_0, num):
    list_1 = copy.deepcopy(list_0)
    list_2 = copy.deepcopy(list_0)
    times = num // len(list_2)
    if times > 1:
        list_2.extend((times-1)*list_1)
    random.seed(36)
    list_2.extend(random.sample(list_2,num-len(list_2)))
    return list_2


def random_del(neb_list):
    seed = random.randint(0,len(neb_list)-1)
    neb_list.pop(seed)
    return neb_list


class CCDataset(Dataset):
    def __init__(self, Data_path, ptid, slide, Mag='5', transforms=None, limit=96, shuffle=False, extd=7):
        self.ptids = ptid
        self.slide = [(ptid, slide)]
#         self.slide = [
#             (ptid, slide) 
#             for ptid in ptids
#             for slide in os.listdir(os.path.join(Data_path, ptid))
#             if limit <= len(glob.glob(os.path.join(Data_path, ptid, slide, Mag, '*')))
#         ]
        
        index = 0
        self.patch = []
        self.label = []
        self.indices = {}
        for i, (ptid, slide) in enumerate(self.slide):
            patches_t = glob.glob(os.path.join(Data_path, ptid, slide, Mag, '*'))
            patches_a = glob.glob(os.path.join(Data_path, ptid, slide, Mag.split('_')[0], '*'))
            
            if len(patches_a) < extd+1:
                patches_a = resampling(patches_a, extd+1)
            
            self.patch.extend(patches_t)
            self.patch.extend(patches_a)
            
            label = data_map[ptid]['patient-label']
            
            self.label.extend([label]*len(patches_t))
            self.label.extend([label]*len(patches_a))
            
            range_t = np.arange(index, index+len(patches_t))
            index += len(patches_t)
            range_a = np.arange(index, index+len(patches_a))
            index += len(patches_a)
            
            nbs = NearestNeighbors(extd+1).fit(list(map(get_loc, patches_a)))
            
            self.indices[(ptid, slide)] = []
            for i in range(len(patches_t)):
                nb_list = list(nbs.kneighbors(get_loc(patches_t[i]).reshape(1,-1),
                                              return_distance=False)[0])[1:]
                for e in range(len(nb_list)-extd):
                    random.seed(666*i+e)
                    nb_list = random_del(nb_list)
                inx = []
                for nl in nb_list:
                    inx.append(range_a[nl])
                self.indices[(ptid, slide)].append([range_t[i]]+inx)

        self.slide = np.array(self.slide)
        self.data_transforms = transforms
        
    def __len__(self):
        return len(self.patch)
    
    def __getitem__(self, index):
        img = Image.open(self.patch[index])
        label = self.label[index]
        if self.data_transforms is not None:
            img = self.data_transforms(img)
        return img, label

    
class DistSlideSampler(DistributedSampler):
    def __init__(self, dataset, padding, seed, shuffle=False):
        super(DistSlideSampler, self).__init__(dataset)
        self.slide = dataset.slide
        self.indices = dataset.indices
        self.padding = padding
        self.seed = hash(seed)
        self.g = torch.Generator()
        
    def __iter__(self):
        self.g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            len(self.slide) - len(self.slide)%self.num_replicas, 
            generator=self.g
        ).tolist()
        for i in indices[self.rank::self.num_replicas]:
            ptid, slide = self.slide[i]
            yield self.get_slide(ptid, slide)
        
    def __len__(self):
        return len(self.slide) // self.num_replicas
    
    def get_slide(self, ptid, slide):
        indice = self.indices[(ptid, slide)]
        patch_num = len(indice)
        np.random.seed(self.seed % (2**32) + self.epoch)
        if patch_num <= self.padding:
            indice = resampling(indice, self.padding)
            return np.array(indice).flatten()
        else:
            random.seed(time.time()*1000000)
            indice = random.sample(indice, self.padding)
            return np.array(indice).flatten()
        
    
class TestDistSlideSampler(DistributedSampler):
    def __init__(self, k, dataset, limit=512, shuffle=False):
        super(TestDistSlideSampler, self).__init__(dataset)
        self.slide = dataset.slide
        self.indices = dataset.indices
        self.limit = limit
        
    def __len__(self):
        return len(self.slide) // self.num_replicas
    
    def __iter__(self):
        slide = self.slide[len(self.slide)%self.num_replicas:]
        for ptid, slide in slide[self.rank::self.num_replicas]:
            yield self.get_slide(ptid, slide)
            
    def get_slide(self, ptid, slide):
        indice = self.indices[(ptid, slide)]
        patch_num = len(indice)
        if patch_num > self.limit:
#             random.seed(666)
#             indice = random.sample(indice, self.limit)
#             random.seed(time.time()*1000000)
            indice = indice[k*self.limit:min(patch_num, (k+1)*self.limit)]
            return np.array(indice).flatten()
        else:
            return np.array(indice).flatten()
    

def fast_collate(batch, memory_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w, h = imgs[0].size[0], imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        numpy_array = np.asarray(img, dtype=np.uint8)
        numpy_array = np.rollaxis(numpy_array, 2)
        tensor[i] += torch.from_numpy(numpy_array.copy())
    return tensor, targets


class data_prefetcher():
    def __init__(self, loader, dataset='train'):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        if dataset=='test2____':
            self.mean = torch.tensor([179.39, 105.45, 168.53]).cuda().view(1,3,1,1)
            self.std = torch.tensor([25.39, 31.86, 19.66]).cuda().view(1,3,1,1)
        else:
            self.mean = torch.tensor([165.65, 100.58, 156.62]).cuda().view(1,3,1,1)
            self.std = torch.tensor([27.72, 28.29, 19.74]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target
    
    
class Attention_Gated(nn.Module):
    def __init__(self, model, pretrain, extd=7):
        super(Attention_Gated, self).__init__()
        self.extd = extd
        self.L = 512
        self.D = 128
        self.K = 1
        
#         self.feature_extractor = pretrainedmodels.inceptionv4(pretrained='imagenet')
        self.feature_extractor = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        
#         self.feature_extractor.last_linear = nn.Linear(1536, self.L)
        self.feature_extractor.fc = nn.Linear(2048, self.L)
        if args.local_rank == 0:
            input_test = torch.randn(1, 3, 224, 224)
            flops, params = profile(self.feature_extractor, inputs=(input_test, ))
            print('FLOPS:', flops)
            print('PARAMS:', params)
#         nn.init.xavier_normal_(self.feature_extractor.last_linear.weight)
        nn.init.xavier_normal_(self.feature_extractor.fc.weight)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.L, 
                                       nhead=8,
                                       activation='gelu'),
            num_layers=2,
            norm=nn.LayerNorm(normalized_shape=self.L, eps=1e-6)
        )
        
        self.inner_attention = nn.Linear(self.L, self.K)
        nn.init.xavier_normal_(self.inner_attention.weight)
        
        self.attention = nn.Linear(self.L, self.K)
        nn.init.xavier_normal_(self.attention.weight)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )
        nn.init.xavier_normal_(self.classifier[0].weight)

    
    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor(x)
        H = H.view((-1, self.extd+1, self.L))
        H = self.encoder(H.transpose(0,1))
        H = H.transpose(0,1)
        
        H = torch.cat([torch.mm(F.softmax(self.inner_attention(h).transpose(0,1), dim=1), h) for h in H], 0)
        
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, H)
        Y_prob = self.classifier(M)
        
        return Y_prob
        
        
class Attention_Gated_Test(Attention_Gated):
    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor(x)
        H = H.view((-1, self.extd+1, self.L))
        H = self.encoder(H.transpose(0,1))
        H = H.transpose(0,1)
        
        H = torch.cat([torch.mm(F.softmax(self.inner_attention(h).transpose(0,1), dim=1), h) for h in H], 0)
        
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, H)
        Y_prob = self.classifier(M)
        
        return Y_prob

    
def get_last_conv_name(net):
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


if __name__ == '__main__':
    args = get_parser()
    
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method=args.init_method
    )
    
    device = torch.device(f"cuda:{args.local_rank}")
    
    model = apex.parallel.convert_syncbn_model(
        Attention_Gated(args.model, True, extd=args.extd)
    ).to(device)

    mg = args.mag.split('_')[0]
    epo = args.epoch
    
    model = amp.initialize(model,opt_level="O0", keep_batchnorm_fp32=None)
    model = DistributedDataParallel(model, delay_allreduce=True)
    model_path = f'../script/checkpoints_{mg}X_{args.sample}_F{args.fold}/comment/{epo}.pt'
    model.load_state_dict(torch.load(model_path))
    
    for vl in val_label:
        val_slide = [f for f in os.listdir(args.path+vl) if 'ipy' not in f]
        for vs in val_slide:
            if os.path.exists(args.path+'/'+vl+'/'+vs+'/'+args.mag+'/'):
                eval_datasets = CCDataset(args.path, 
                                          vl, 
                                          vs,
                                          Mag=args.mag, 
                                          transforms=transforms.Compose([
#                                               transforms.CenterCrop(384),
                                              transforms.Resize(299),
                                              ]),
                                          limit=2, 
                                          shuffle=False,
                                          extd=args.extd)

                memory_format = torch.contiguous_format
                collate_fn = lambda b: fast_collate(b, memory_format)

                numslide = len(os.listdir(args.path+'/'+vl+'/'+vs+'/'+args.mag+'/'))
                times = numslide//int(args.test_limit)+1
                print('Drawing CAM:', vl, vs)
                print('Num of patches:', len(os.listdir(args.path+'/'+vl+'/'+vs+'/'+args.mag+'/')))
                print('Loops:', str(times))

                for k in range(times):
                    if min(numslide, (k+1)*int(args.test_limit))-k*int(args.test_limit) > 1:
                        print('Sampling patch:', k*int(args.test_limit), 'to', min(numslide, (k+1)*int(args.test_limit)))
                        sampler = TestDistSlideSampler(k, eval_datasets, limit=args.test_limit)
                        eval_loader = DL(eval_datasets, 
                                         batch_sampler=sampler,
                                         num_workers=16,
                                         pin_memory=True,
                                         collate_fn=collate_fn)

                        layer_name = get_last_conv_name(model.module.feature_extractor)
        #                 print('Last conv layer name:', layer_name)
                        model.eval()

                        prefetcher = data_prefetcher(eval_loader)
                        patches, label = prefetcher.next()
                        index = 0

                        if os.path.exists(args.save_path):
                            if os.path.exists(f'{args.save_path}/CAM_{args.fold}'):
                                pass
                            else:
                                os.makedirs(f'{args.save_path}/CAM_{args.fold}')
                        else:
                            os.makedirs(args.save_path)
                            os.makedirs(f'{args.save_path}/CAM_{args.fold}')

                        while patches is not None:
                            ptid, slide = sampler.slide[index]
                            if os.path.exists(f'{args.save_path}/CAM_{args.fold}/{ptid}'):
                                if os.path.exists(f'{args.save_path}/CAM_{args.fold}/{ptid}/{slide}'):
                                    pass
                                else:
                                    os.makedirs(f'{args.save_path}/CAM_{args.fold}/{ptid}/{slide}')
                            else:
                                os.makedirs(f'{args.save_path}/CAM_{args.fold}/{ptid}')
                                os.makedirs(f'{args.save_path}/CAM_{args.fold}/{ptid}/{slide}')
                            idxs = sampler.get_slide(ptid, slide)
                            path = [eval_datasets.patch[i] for i in idxs]
    #                         print(ptid, slide, len(idxs))

                            model.zero_grad()

                            handler = []
                            feature = None
                            gradient = None

                            def get_feature_hook(module, input, output):
                                global feature
                                feature = output

                            def get_grads_hook(module, input, output):
                                global gradient
                                gradient = output[0]

                            for (name, module) in \
                                model.module.feature_extractor.named_modules():
                                if name == layer_name:
                                    handler.append(module.register_forward_hook(get_feature_hook))
                                    handler.append(module.register_backward_hook(get_grads_hook))

                    #         handler.append(
                    #             model.module.feature_extractor_part1.features\
                    #             .register_forward_hook(get_feature_hook))
                    #         handler.append(
                    #             model.module.feature_extractor_part1.features\
                    #             .register_backward_hook(get_grads_hook))

                            Y_prob = model.forward(patches)
                            Y_prob.backward()
                    #         print(feature.shape)
                            for i in range(len(idxs)):
                                f = feature[i].cpu().data.numpy() # 256 * 8 * 8
                                g = gradient[i].cpu().data.numpy() # 256 * 8 * 8
                                weight = np.mean(g, axis=(1, 2)) # 256, 

                                cam = f * weight[:, np.newaxis, np.newaxis] # 256 * 8 * 8
                                cam = np.sum(cam, axis=0) # 256, 
                                cam -= np.min(cam)
                                cam /= np.max(cam)
                                cam = cv2.resize(cam, (299, 299))

                                img = Image.open(path[i])
                                img = img_transform(img)
                                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

                                heatmap = cam*0.6 + img * 0.4
#                                 heatmap = np.vstack((heatmap, img))    # Top
#                                 heatmap = np.hstack((heatmap, img))    # Left
#                                 heatmap = np.vstack((img, heatmap))    # Bottom
                                heatmap = np.hstack((img, heatmap))    # Right

                                file_name = '{}_CAM.jpeg'.format(os.path.basename(path[i]).split('.')[0])
                                cv2.imwrite(f'{args.save_path}/CAM_{args.fold}/{ptid}/{slide}/{file_name}', heatmap)

                            for h in handler:
                                h.remove()
                            patches, label = prefetcher.next()
                            index += 1
                    else:
                        print(f'Remain 1 patch: {args.path}/{vl}/{vs}/{args.mag}/')
            else:
                pass

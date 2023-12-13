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
import argparse
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
import pandas as pd
import copy
import shutil
from thop import profile
import sys
from torchvision.utils import save_image
from torch.autograd import Variable
from models import Generator
from datasets import ImageDataset
from scipy import stats
import datetime
import math

plt.switch_backend('Agg')


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


class MVIDataset(Dataset):
    def __init__(self, Data_path, ptid, slide, Mag='5', transforms=None, limit=96, shuffle=False, extd=7):
        self.ptids = ptid
        self.slide = [(ptid, slide)]
        
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
        name = '|'.join(self.patch[index].split('/')[3:])
        img = Image.open(self.patch[index])
        label = self.label[index]
        if self.data_transforms is not None:
            img = self.data_transforms(img)
        return img, label, name

    
class DistSlideSampler(DistributedSampler):
    def __init__(self, k, dataset, padding, seed, shuffle=False):
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
    names = [name[2] for name in batch]
    w, h = imgs[0].size[0], imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        numpy_array = np.asarray(img, dtype=np.uint8)
        numpy_array = np.rollaxis(numpy_array, 2)
        tensor[i] += torch.from_numpy(numpy_array.copy())
    return tensor, targets, names


class data_prefetcher():
    def __init__(self, loader, dataset='train'):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([145.28, 85.00, 147.10]).cuda().view(1,3,1,1)
        self.std = torch.tensor([27.72, 28.29, 19.74]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_name = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_name = None
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
        name = self.next_name
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target, name
    
    
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
        
        return A, Y_prob
        
        
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
        
        return A, Y_prob

            
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    return rt


def gather_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    var_list = [torch.zeros_like(rt) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(var_list, rt, async_op=False)
    return [i.item() for i in var_list]


def eval_model(args, dataloader, model, k):
    model.eval()
    all_labels = []
    all_values = []
    train_loss = 0
    
    prefetcher = data_prefetcher(dataloader)
    patches, label, name = prefetcher.next()
    index = 0
    prob_df = pd.DataFrame({'Center':[], 'Neighb':[], 'Prob':[]})
    while patches is not None:
        index += 1
        label = label[0].float()
        
        with torch.no_grad():
            A, Y_prob= model.forward(patches)
#             print('-'*30)
            prob = list(np.array(A[0].cpu()))
#             print(prob)
            center = []
            neighb = []
            for i in range(0, len(name), args.extd+1):
                center.append(name[i])
                neighb.append(name[i+1:i+args.extd+1])
#             print(center)
#             print(neighb)
            prob_df = prob_df.append(pd.DataFrame({'Center':center, 'Neighb':neighb, 'Prob':prob}))
            Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        
        all_labels.extend(gather_tensor(label))
        all_values.extend(gather_tensor(Y_prob[0][0]))
        
        patches, label, name = prefetcher.next()
            
    if args.local_rank == 0:
        print(len(all_labels))
        prob_df = prob_df.reset_index(drop=True)
        prob_df.to_excel(f'./df_middle/prob_df_{k}.xlsx', index=None)
        
    return

    
    
def get_cm(AllLabels, AllValues):
    fpr, tpr, threshold = roc_curve(AllLabels, AllValues, pos_label=1)
    Auc = auc(fpr, tpr)
    m = t = 0

    for i in range(len(threshold)):
        if tpr[i] - fpr[i] > m :
            m = abs(-fpr[i]+tpr[i])
            t = threshold[i]
    AllPred = [int(i>=t) for i in AllValues]
    Acc = sum([AllLabels[i] == AllPred[i] for i in range(len(AllPred))]) / len(AllPred)

    Pos_num = sum(AllLabels)
    Neg_num = len(AllLabels) - Pos_num
    cm = confusion_matrix(AllLabels, AllPred)
    print("[AUC/{:.4f}] [Threshold/{:.4f}] [Acc/{:.4f}]".format(Auc, t,  Acc))
    print("{:.2f}% {:.2f}%".format(cm[0][0]/ Neg_num * 100, cm[0][1]/Neg_num * 100))
    print("{:.2f}% {:.2f}%".format(cm[1][0]/ Pos_num * 100, cm[1][1]/Pos_num * 100))
    
    return Auc, Acc


def get_auc(ture, pred):
    fpr, tpr, thresholds = metrics.roc_curve(ture, pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def save_roc(ture, pred, imgn):
    fpr, tpr, thresholds = metrics.roc_curve(ture, pred, pos_label=1)
    plt.cla()
    plt.plot(fpr,tpr)
    plt.savefig(f'{imgn}.jpg')
    return


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of multiple Instance learning')
    parser.add_argument('--path', default='None', type=str)
    parser.add_argument('--mag', default='5', type=str)
    parser.add_argument('--sample', default='All', type=str)
    parser.add_argument('--fold', default='0', type=str)
    parser.add_argument('--epo', default='5', type=str)
    parser.add_argument('--model', default=18, type=str)
    parser.add_argument('--test_limit', default=50, type=int)
    parser.add_argument('--extd', default=7, type=int)
    parser.add_argument('--option', default='TEST', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init_method', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    
    with open('./pat_labels.json3') as f:
        data_map = json.load(f)

    test_transform = transforms.Compose([
                transforms.CenterCrop(384),
                transforms.Resize(299),
            ])
    
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method=args.init_method
    )
    
    if not os.path.exists('./df_final/'):
        os.mkdir('./df_final/')
        os.mkdir('./df_final/X10/')
        os.mkdir('./df_final/X20/')
        os.mkdir('./df_final/X40/')
    
    data_path = args.path
    pat_slide_all = pd.read_excel('pat_slide_all.xlsx')
    avlb_slide = os.listdir(data_path)
    
    avlb_slide_df = pat_slide_all[pat_slide_all['slide'].isin(avlb_slide)]
    all_id_list = list(set(avlb_slide_df['pat'].tolist()))
    
    random.seed(int(args.sample.split('_')[1]))
    random.shuffle(all_id_list)
    
    F0_valid_pat = all_id_list[int(0.8*len(all_id_list)):]
    test0_df = avlb_slide_df[avlb_slide_df['pat'].isin(F0_valid_pat)]
    test0_label = list(set(test0_df['slide'].tolist()))
    
    old_neg = list(set(pd.read_excel('old_slide_all.xlsx')['old_slide'].tolist())&set(test0_label))
    test0_label = list(set(test0_label)-set(old_neg))
    
    hotmap_pat = list(set(test0_label)-set([f.split('_')[3] for f in os.listdir(f'./df_final/X{args.mag}/')]))
    for hp in hotmap_pat:
        hotmap_slide = os.listdir(args.path+hp)
        for hs in hotmap_slide:
            try:
                if not os.path.exists('./df_middle/'):
                    os.mkdir('./df_middle/')
                else:
                    shutil.rmtree('./df_middle/')
                    os.mkdir('./df_middle/')
                eval_datasets = MVIDataset(args.path, 
                                           hp, 
                                           hs,
                                           Mag=args.mag, 
                                           transforms=transforms.Compose([
                                                transforms.CenterCrop(384),
                                                transforms.Resize(299),
                                            ]),
                                           limit=1,
                                           shuffle=False,
                                           extd=args.extd)

                memory_format = torch.contiguous_format
                collate_fn = lambda b: fast_collate(b, memory_format)

                numslide = len(os.listdir(args.path+'/'+hp+'/'+hs+'/'+args.mag+'/'))
                times = numslide//int(args.test_limit)+1
                print('Num of all clusters:', len(os.listdir(args.path+'/'+hp+'/'+hs+'/'+args.mag+'/')))
                print('Loops:', str(time))

                for k in range(times):
                    print('Sampling cluster:', k*int(args.test_limit), 'to', min(numslide, (k+1)*int(args.test_limit)))
                    sampler = TestDistSlideSampler(k, eval_datasets, limit=args.test_limit)
                    eval_loader = DL(eval_datasets, 
                                     batch_sampler=sampler,
                                     num_workers=16,
                                     pin_memory=True,
                                     collate_fn=collate_fn,
                                     shuffle=False)

            #         if args.local_rank == 0:
            #             print(' slide number:', len(eval_datasets.slide))
            #             print(' patches number:', len(eval_datasets))

                    device = torch.device(f"cuda:{args.local_rank}")

                    model = apex.parallel.convert_syncbn_model(
                        Attention_Gated(args.model, True, extd=args.extd)
                    ).to(device)

                    model = amp.initialize(model,opt_level="O0", keep_batchnorm_fp32=None)
                    model = DistributedDataParallel(model, delay_allreduce=True)

                    mg = args.mag.split('_')[0]
                    option = args.option
                    epo = args.epo
                    print('-'*30)
                    print('Testing mag:', mg)
                    print('Model epo:', epo)
                    print('-'*30)
                    path = f'./checkpoints_{mg}X_{args.sample}_F{args.fold}/comment/{epo}.pt'
                    model.load_state_dict(torch.load(path))
                    eval_model(args, eval_loader, model, k)

                df_0 = pd.read_excel('./df_middle/prob_df_0.xlsx')
                for k in range(1,times):
                    df_k = pd.read_excel(f'./df_middle/prob_df_{k}.xlsx')
                    df_0 = df_0.append(df_k)

                df_0.to_excel(f'./df_final/X{args.mag}/prob_df_{args.mag}X_{hp}_{hs}.xlsx')
            except Exception:
                pass
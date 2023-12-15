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


class CCDataset(Dataset):
    def __init__(self, Data_path, ptids, Mag='5', transforms=None, limit=96, shuffle=False, extd=7):
        self.ptids = ptids
        self.slide = [
            (ptid, slide) 
            for ptid in ptids
            for slide in os.listdir(os.path.join(Data_path, ptid))
            if limit <= len(glob.glob(os.path.join(Data_path, ptid, slide, Mag, '*')))
        ]
        
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
    def __init__(self, dataset, limit=512, shuffle=False):
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
            random.seed(666)
            indice = random.sample(indice, self.limit)
            random.seed(time.time()*1000000)
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

            
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    return rt


def gather_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    var_list = [torch.zeros_like(rt) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(var_list, rt, async_op=False)
    return [i.item() for i in var_list]


def eval_model(args, dataloader, model):
    model.eval()
    all_labels = []
    all_values = []
    train_loss = 0
    
    prefetcher = data_prefetcher(dataloader)
    patches, label = prefetcher.next()
    index = 0
    while patches is not None:
        index += 1
        label = label[0].float()
        
        with torch.no_grad():
            Y_prob= model.forward(patches)
            Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)

            J = -1.*(
                label*torch.log(Y_prob)+
                (1.-label)*torch.log(1.-Y_prob)
            )
        
        reduced_loss = reduce_tensor(J.data)
        
        train_loss += reduced_loss.item()
        all_labels.extend(gather_tensor(label))
        all_values.extend(gather_tensor(Y_prob[0][0]))
        
        patches, label = prefetcher.next()
            
            
    if args.local_rank == 0:
        print(len(all_labels))
        all_labels = np.array(all_labels)
        Loss = train_loss / len(all_labels)
        AUC, Acc = get_cm(all_labels, all_values)
        
    return all_labels, all_values

    
    
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
    parser.add_argument('--mag', default='10', type=str)
    parser.add_argument('--model_name', default='All', type=str)
    parser.add_argument('--path_simc', default='/data_path/', type=str)
    parser.add_argument('--epo', default='5', type=str)
    parser.add_argument('--model', default='inceptionv3', type=str)
    parser.add_argument('--extd', default=7, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--init_method', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    
    with open('./pat_labels.json3') as f:
        data_map = json.load(f)
        
    save_file = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d-%H-%M-%S')
    os.mkdir(f'./{save_file}_{args.model_name}_X{args.mag}/')
        
    pat_slide_all = pd.read_excel('pat_slide_all.xlsx')
    test_path = {
        'train': '../train/',
        'valid': '../valid/',
        'test': '../test/'
                }
        
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
    
    model = amp.initialize(model,opt_level="O0", keep_batchnorm_fp32=None)
    model = DistributedDataParallel(model, delay_allreduce=True)

    # option = 'train'
    option = 'valid'
    # option = 'test'

    if option == 'valid' or option == 'train' or option == 'trainval':
        pat_slide_all = pd.read_excel('pat_slide_all.xlsx')
        avlb_slide = os.listdir(test_path[option])
        avlb_slide_df = pat_slide_all[pat_slide_all['slide'].isin(avlb_slide)]
        all_id_list = list(set(avlb_slide_df['pat'].tolist()))
        random.seed(int(args.model_name.split('_')[1]))
        random.shuffle(all_id_list)
        F0_valid_pat = all_id_list[int(0.8*len(all_id_list)):]
        KF_all_id = list(set(all_id_list)-set(F0_valid_pat))
        random.shuffle(KF_all_id)
    else:
        test_label = os.listdir(test_path[option])
        test_label.sort()


    test_transform = transforms.Compose([
                transforms.CenterCrop(384),
                transforms.Resize(299),
            ])

    for fd in range(5):
        os.mkdir(f'./{save_file}_{args.model_name}_X{args.mag}/F{fd}/')
        if option == 'train':
            valid_pat = KF_all_id[int(0.2*len(KF_all_id)*fd):int(0.2*len(KF_all_id)*(fd+1))]
            train_pat = list(set(KF_all_id)-set(valid_pat))

            train_df = avlb_slide_df[avlb_slide_df['pat'].isin(train_pat)]
            train_label = list(set(train_df['slide'].tolist()))
            valid_df = avlb_slide_df[avlb_slide_df['pat'].isin(valid_pat)]
            val_label = list(set(valid_df['slide'].tolist()))

            #################
            old_neg = list(set(pd.read_excel('old_slide_all.xlsx')['old_slide'].tolist())&(set(train_label)|set(val_label)))
            train_label = list(set(train_label)-set(old_neg))
            val_label = list(set(val_label)-set(old_neg))
            #################

            train_label = list(set(train_label)-set(val_label))

            test_label = train_label
            test_label.sort()
        elif option == 'valid':
            valid_pat = KF_all_id[int(0.2*len(KF_all_id)*fd):int(0.2*len(KF_all_id)*(fd+1))]
            train_pat = list(set(KF_all_id)-set(valid_pat))

            train_df = avlb_slide_df[avlb_slide_df['pat'].isin(train_pat)]
            train_label = list(set(train_df['slide'].tolist()))
            valid_df = avlb_slide_df[avlb_slide_df['pat'].isin(valid_pat)]
            val_label = list(set(valid_df['slide'].tolist()))

            #################
            old_neg = list(set(pd.read_excel('old_slide_all.xlsx')['old_slide'].tolist())&(set(train_label)|set(val_label)))
            train_label = list(set(train_label)-set(old_neg))
            val_label = list(set(val_label)-set(old_neg))
            #################

            train_label = list(set(train_label)-set(val_label))

            test_label = val_label
            test_label.sort()
        elif option == 'trainval':
            trainval_df = avlb_slide_df[avlb_slide_df['pat'].isin(KF_all_id)]
            trainval_label = list(set(trainval_df['slide'].tolist()))

            #################
            old_neg = list(set(pd.read_excel('old_slide_all.xlsx')['old_slide'].tolist())&(set(trainval_label)))
            trainval_label = list(set(trainval_label)-set(old_neg))
            #################

            test_label = trainval_label
            test_label.sort()
        else:
            pass

        eval_datasets = CCDataset(test_path[option], 
                                  test_label, 
                                  Mag=args.mag,
                                  transforms=transforms.Compose([
                                      transforms.CenterCrop(384),
                                      transforms.Resize(299),
                                      ]),
                                  limit=2,
                                  shuffle=False,
                                  extd=args.extd)

        memory_format = torch.contiguous_format
        collate_fn = lambda b: fast_collate(b, memory_format)
        sampler = TestDistSlideSampler(eval_datasets, limit=50)
        eval_loader = DL(eval_datasets, 
                         batch_sampler=sampler,
                         num_workers=16,
                         pin_memory=True,
                         collate_fn=collate_fn,
                         shuffle=False)

        if args.local_rank == 0:
            print('='*30)
            print('Slide number:', len(eval_datasets.slide))
            print('Patches number:', len(eval_datasets))

        mg = args.mag.split('_')[0]
#         epo = args.epo
        print('-'*30)
        epomax = np.max([int(i.split('.')[0]) for i in os.listdir(f'./checkpoints_{mg}X_{args.model_name}_F{fd}/comment/') if i[-2:]=='pt'])+1

        EpochList = []
        PatAucMinList = []
        PatAucMeanList = []
        PatAucMaxList = []

        for epoch in range(1, epomax):
            epo = str(epoch)

            model.load_state_dict(torch.load(f'./checkpoints_{mg}X_{args.model_name}_F{fd}/comment/{epo}.pt'))
            print(f'Testing: Mag-{mg}X-Epo{epo}')
            print(f'Model: {option}-X{mg}-{args.model_name}-{epo}')
            print('-'*30)

            all_labels, all_values = eval_model(args, eval_loader, model)
            if args.local_rank == 0:
                import pandas as pd
                result = pd.DataFrame({
                    'Pat':[p.split('-')[-1] for p in eval_datasets.slide[:,0]],
                    'tile_id':eval_datasets.slide[:,0],
                    'Slide':eval_datasets.slide[:,1],
                    'Label':all_labels,
                    'Value':all_values
                })
                result.to_csv(f'./{save_file}_{args.model_name}_X{args.mag}/F{fd}/result-{option}-X{args.mag}-{epo}.csv')

#                 result = pd.read_csv(f'./2022-06-20-09-24-15_20226666KF_299_inveptionv3_X10/F{fd}/result-{option}-X{args.mag}-{epo}.csv')

                pat_slide = pd.read_excel('pat_slide_all.xlsx', names=['no', 'pat_id', 'tile_id'])
                result = pd.merge(result, pat_slide, how='left', on='tile_id')
                result['type'] = result['tile_id'].apply(lambda x:x.split('-')[1])
                result = result[['pat_id','Value','Label','type']]
                result['Label'] = result['Label'].astype('int')

                result_mean = result.groupby('pat_id').agg({'Value':'mean', 'Label':'mean'}).reset_index()
                Pat_AUC_mean = get_auc(result_mean['Label'], result_mean['Value'])

                result_min = result.groupby('pat_id').agg({'Value':'min', 'Label':'mean'}).reset_index()
                Pat_AUC_min = get_auc(result_min['Label'], result_min['Value'])

                result_max = result.groupby('pat_id').agg({'Value':'max', 'Label':'mean'}).reset_index()
                Pat_AUC_max = get_auc(result_max['Label'], result_max['Value'])

                print('Pat AUC-mean:', round(Pat_AUC_mean,4))
                print('Pat AUC-min:', round(Pat_AUC_min,4))
                print('Pat AUC-max:', round(Pat_AUC_max,4))

                save_roc(all_labels, all_values, 'Slide_level')
                save_roc(result_max['Label'], result_max['Value'], 'Pat_level')

                print('Slide prediction mean:', round(np.mean(all_values),4))
                print('Slide prediction median:', round(np.median(all_values),4))
                n, min_max, mean, var, skew, kurt = stats.describe(all_values)
                std = math.sqrt(var)
                CI = stats.norm.interval(0.95, loc=mean, scale=std)
                print('Slide prediction 0.95 CI:', '['+str(round(CI[0], 4))+', '+str(round(CI[1], 4))+']')

                EpochList.append(epo)
                PatAucMinList.append(Pat_AUC_min)
                PatAucMeanList.append(Pat_AUC_mean)
                PatAucMaxList.append(Pat_AUC_max)

        RESULTS_DF = pd.DataFrame({
            'EPOCH': EpochList,
            f'{option}AucMean': PatAucMeanList,
            f'{option}AucMax': PatAucMaxList,
            f'{option}AucMin': PatAucMinList})

        RESULTS_DF.to_excel(f'./{save_file}_{args.model_name}_X{args.mag}/{option}_F{fd}_epochs_detail.xlsx', index=None)

        test_dataframe = pd.DataFrame({'test_label': test_label})
        test_dataframe.to_excel(f'./{save_file}_{args.model_name}_X{args.mag}/test_label_{option}_F{fd}.xlsx', index=None)
        

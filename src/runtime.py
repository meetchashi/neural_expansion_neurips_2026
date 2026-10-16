"""Shared loaders for the portable reproduction entry points."""
import importlib
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torchvision import datasets, transforms, models
from src.datasets import training_split

IMAGENETTE_CLASSES = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_for(value='auto'):
    return torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if value == 'auto' else value)

def load_custom(config_path, checkpoint, device):
    config = yaml.safe_load(Path(config_path).read_text())
    m = config['model']
    model = getattr(importlib.import_module(m.get('module','src.models')+'.'+m['file']),m['name'])(**m.get('args',{}))
    if checkpoint: model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    return model.to(device).eval(), config

def load_run(args):
    seed_all(args.seed)
    device = device_for(args.device)
    if args.pretrained:
        if args.pretrained in ('resnet20','vgg16_bn'):
            model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_'+args.pretrained, pretrained=True)
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
            train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform)
            test = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform)
            config = {'dataset': {'name':'CIFAR10','transform':{'normalize':{'mean':[0.5]*3,'std':[0.5]*3}}}}
        else:
            if args.pretrained in ('vgg16','resnet50'):
                model = getattr(models,args.pretrained)(weights='IMAGENET1K_V1')
            else:
                import timm
                model = timm.create_model({'vit_base':'vit_base_patch16_224','deit_small':'deit_small_patch16_224'}[args.pretrained],pretrained=True)
            norm = {'mean':[0.485,0.456,0.406],'std':[0.229,0.224,0.225]}
            transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(**norm)])
            train, test = [datasets.ImageFolder(str(Path(args.data_path)/s),transform=transform) for s in ('train','val')]
            expected = ['n01440764','n02102040','n02979186','n03000684','n03028079','n03394916','n03417042','n03425413','n03445777','n03888257']
            if train.classes != expected or test.classes != expected: raise ValueError('ImageNette synset folders do not match the expected 10 classes')
            train, test = MappedLabels(train), MappedLabels(test)
            config = {'dataset':{'name':'ImageNette','transform':{'normalize':norm}}}
        return model.to(device).eval(), config, train, test, device
    model, config = load_custom(args.config, args.checkpoint, device)
    if args.data_path: config['dataset']['data_dir']=args.data_path
    train, _, test = training_split(config)
    return model, config, train, test, device

class MappedLabels(torch.utils.data.Dataset):
    def __init__(self,dataset): self.dataset=dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self,i):
        x,y=self.dataset[i];return x,IMAGENETTE_CLASSES[y]

def sample_indices(dataset, per_class, seed):
    pools={}
    for i in range(len(dataset)): pools.setdefault(int(dataset[i][1]),[]).append(i)
    rng=np.random.RandomState(seed)
    return [int(i) for ids in pools.values() for i in (rng.choice(ids,min(per_class,len(ids)),replace=False) if per_class else ids)]

def save_metadata(args, config):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'run.json').write_text(json.dumps({'arguments':vars(args),'config':config,'torch':torch.__version__,'numpy':np.__version__},indent=2,default=str)+'\n')
    return out

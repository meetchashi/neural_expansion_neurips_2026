"""Coverage, directional stability, linearity, attack and gradient-projection experiments.

Numerical conventions are documented in docs/REPRODUCIBILITY.md.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from src.analysis.gi_calculation import compute_GI
from src.runtime import load_run, sample_indices, save_metadata


def gi(model, anchors, targets, label, args, device):
    return compute_GI(model, anchors, targets, False, label, step_size=args.step_size,
                      max_eps=args.max_eps, device=device)[0]


def coverage_row(model, image, label, pools, args, device, sample_id, train):
    with torch.no_grad(): pred = int(model(image.unsqueeze(0).to(device)).argmax(1))
    ids = pools.get(pred, [])
    if args.pool_size:
        ids = np.random.choice(len(ids), min(args.pool_size,len(ids)),replace=False) if ids else []
        images = [pools[pred][i] for i in ids]
    else: images = ids
    values=[]
    for start in range(0,len(images),args.batch_size):
        anchors=torch.stack([train[i][0] for i in images[start:start+args.batch_size]]).to(device)
        target=image.to(device).unsqueeze(0).expand_as(anchors)
        values.extend(gi(model,anchors,target,pred,args,device).tolist())
    row={'sample_id':sample_id,'true_label':label,'predicted_class':pred,
         'correct':pred==label,'pairs':len(values),'coverage':100*float(np.mean(np.array(values)>1)) if values else np.nan,
         'mean_gi':float(np.mean(values)) if values else np.nan,
         'status':'measured' if values else 'no_training_pool'}
    return row,values


def run_coverage(args, ood=False, attacks=False):
    model,config,train,test,device=load_run(args)
    out=save_metadata(args,config)
    if ood:
        test=datasets.CIFAR10(args.ood_data_path,train=False,download=True,transform=transforms.Compose([
            transforms.Grayscale(1),transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])]))
    pools={}
    for i in range(len(train)):
        x,y=train[i];pools.setdefault(y,[]).append(i)
    rows=[]
    for idx in sample_indices(test,args.k,args.seed):
        x,y=test[idx]
        if attacks: x=attack(model,x.to(device),y,config,args).cpu()
        row,values=coverage_row(model,x,y,pools,args,device,idx,train)
        rows.append(row)
        if args.save_pairs: np.save(out/f'gi_{idx}.npy',np.array(values))
        # Flush after each test example so long jobs retain completed measurements.
        pd.DataFrame(rows).to_csv(out/'coverage.csv',index=False)
    df=pd.DataFrame(rows)
    measured=df[df['status']=='measured']
    measured.groupby('correct')['coverage'].agg(['count','mean','std','median']).to_csv(out/'summary.csv')
    if len(measured):
        measured.boxplot(column='coverage',by='correct');plt.ylabel('Training-pair GI > 1 (%)');plt.suptitle('');plt.tight_layout();plt.savefig(out/'coverage.png');plt.close()
    if config['dataset'].get('variant') in ('lt1','lt2'):
        from src.datasets import LT_COUNTS
        counts=LT_COUNTS[config['dataset']['variant']]
        # Bands are those explicitly listed in the paper, including its LT2 class-6 assignment.
        bands=['H']*3+['M']*(4 if config['dataset']['variant']=='lt1' else 3)
        bands+=['L']*(10-len(bands))
        df['true_band']=df.true_label.map(dict(enumerate(bands)))
        df['predicted_band']=df.predicted_class.map(dict(enumerate(bands)))
        pd.crosstab(df.loc[~df.correct,'true_band'],df.loc[~df.correct,'predicted_band'],normalize='index').mul(100).to_csv(out/'misclassification_by_frequency.csv')
        df.groupby(['true_band','predicted_band']).coverage.mean().to_csv(out/'coverage_by_frequency.csv')


def feature_function(model):
    if hasattr(model,'extract_features'): return lambda x:model.extract_features(x)[-2]
    # Keep the original 1000-way classifier for decisions; hook only for Jacobian features.
    if hasattr(model,'forward_features'):
        return lambda x:model.forward_head(model.forward_features(x),pre_logits=True)
    final = getattr(model, 'fc', None)
    if final is None and hasattr(model, 'classifier'):
        final = model.classifier[-1]
    if final is not None:
        def features(x):
            captured = []
            handle = final.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
            try:
                model(x)
            finally:
                handle.remove()
            return captured[0]
        return features
    raise ValueError('Unsupported feature extractor')


def jacobian_svd(model,x):
    phi=feature_function(model)
    z=x.detach().clone().requires_grad_(True)
    features=phi(z.unsqueeze(0)).flatten()
    jac=torch.stack([torch.autograd.grad(features[i],z,retain_graph=True)[0].flatten() for i in range(len(features))])
    # Full right-singular basis matters for the nullspace. Huge image dimensions use the thin basis.
    full=jac.shape[1]<=4096
    u,s,vh=torch.linalg.svd(jac,full_matrices=full)
    return phi,jac.detach(),s.detach(),vh.detach()


def run_directional(args,mode):
    model,config,train,test,device=load_run(args);out=save_metadata(args,config);rows=[]
    for idx in sample_indices(test,args.k,args.seed):
        x,y=test[idx];x=x.to(device)
        phi,jac,sv,vh=jacobian_svd(model,x)
        scale=args.scale if args.scale else 2*np.sqrt(x.numel())
        indices=np.unique(np.linspace(0,len(vh)-1,min(args.directions,len(vh)),dtype=int))
        if mode=='alignment':
            grad=torch.autograd.functional.jacobian(lambda z:F.cross_entropy(model(z.unsqueeze(0)),torch.tensor([y],device=device)),x).flatten()
            other=train[(idx+1)%len(train)][0].to(device);delta=(other-x).flatten()
            for kind,vector in [('gradient',grad),('image_difference',delta)]:
                energy=(vh@vector).square().cumsum(0)/vector.square().sum().clamp_min(1e-12)
                for i in indices: rows.append({'sample_id':idx,'kind':kind,'rsv_index':int(i),'cumulative_energy':float(energy[i])})
        else:
            with torch.no_grad(): ref=int(model(x.unsqueeze(0)).argmax(1));original=phi(x.unsqueeze(0))
            kinds=['rsv','random'] if mode=='directional' else ['rsv']
            for kind in kinds:
                for i in indices:
                    v=vh[i] if kind=='rsv' else F.normalize(torch.randn(x.numel(),device=device),dim=0)
                    direction=v.reshape_as(x)*scale
                    if mode=='directional':
                        value=float(gi(model,x.unsqueeze(0),(x+direction).unsqueeze(0),ref,args,device)[0])
                    else:
                        tangent=jac@(v*scale);value=args.max_alpha
                        with torch.no_grad():
                            for alpha in np.linspace(0,args.max_alpha,args.alpha_steps):
                                actual=phi((x+alpha*direction).unsqueeze(0))
                                approx=original+alpha*tangent.unsqueeze(0)
                                if F.cosine_similarity(actual,approx).item()<args.cosine_threshold:
                                    value=float(alpha);break
                    rows.append({'sample_id':idx,'label':y,'kind':kind,'rsv_index':int(i),'singular_value':float(sv[i]) if i<len(sv) else 0.,'value':value})
        pd.DataFrame(rows).to_csv(out/'measurements.csv',index=False)
    df=pd.DataFrame(rows)
    value='cumulative_energy' if mode=='alignment' else 'value'
    for kind,group in df.groupby('kind'):
        plt.plot(group.groupby('rsv_index')[value].mean(),label=kind)
    plt.xlabel('Right singular vector index');plt.ylabel(value);plt.legend();plt.tight_layout();plt.savefig(out/'profile.png');plt.close()


def run_trajectories(args):
    model,config,train,test,device=load_run(args);out=save_metadata(args,config)
    rows=[]
    for idx in sample_indices(test,args.k,args.seed):
        x,y=test[idx]
        anchors=[i for i in range(len(train)) if train[i][1]==y][:args.pool_size or 1]
        for anchor_idx in anchors:
            anchor=train[anchor_idx][0].to(device);delta=x.to(device)-anchor
            with torch.no_grad():
                for eps in np.arange(0,args.max_eps+args.step_size/2,args.step_size):
                    pred=int(model((anchor+eps*delta).unsqueeze(0)).argmax(1))
                    rows.append({'sample_id':idx,'anchor_id':anchor_idx,'epsilon':float(eps),'predicted_class':pred,'true_label':y})
    df=pd.DataFrame(rows);df.to_csv(out/'trajectories.csv',index=False)
    for (idx,anchor),g in df.groupby(['sample_id','anchor_id']):
        plt.step(g.epsilon,g.predicted_class,where='post');plt.axvline(1,color='gray',linestyle='--');plt.xlabel('epsilon (test image at 1)');plt.ylabel('Predicted class');plt.tight_layout();plt.savefig(out/f'path_{idx}_{anchor}.png');plt.close()


def attack(model,x,y,config,args):
    """Paper hyperparameters in pixel [0,1] space; GI still uses normalized inputs.

    CW is an untargeted L2 tanh-parameterized optimization with fixed c (no binary search).
    The PNG-based source-compatible pipeline uses a separate targeted CE/MSE objective.
    """
    norm=config['dataset']['transform']['normalize']
    mean=torch.tensor(norm['mean'],device=x.device)[:,None,None]
    std=torch.tensor(norm['std'],device=x.device)[:,None,None]
    pixel=(x*std+mean).clamp(0,1).detach();label=torch.tensor([y],device=x.device)
    logits=lambda z:model(((z-mean)/std).unsqueeze(0))
    if args.attack=='cw':
        w=torch.atanh((pixel*2-1).clamp(-0.999999,0.999999)).detach().requires_grad_(True)
        opt=torch.optim.Adam([w],lr=0.01)
        best=pixel.clone();best_l2=float('inf');last=pixel
        for _ in range(args.attack_steps or 1000):
            adv=(w.tanh()+1)/2;out=logits(adv)
            other=out.clone();other[0,y]=-torch.inf
            loss=(adv-pixel).square().sum()+0.01*(out[0,y]-other.max()).clamp_min(0)
            opt.zero_grad();w.grad=torch.autograd.grad(loss,w)[0];opt.step()
            last=adv.detach();distance=float((adv-pixel).square().sum())
            if int(out.argmax(1))!=y and distance<best_l2:best=last.clone();best_l2=distance
        adv=best if best_l2<float('inf') else last
    else:
        adv=(pixel+torch.empty_like(pixel).uniform_(-0.03,0.03)).clamp(0,1) if args.attack=='pgd' else pixel.clone()
        for _ in range(args.attack_steps or (30 if args.attack=='pgd' else 10)):
            adv=adv.detach().requires_grad_(True)
            grad=torch.autograd.grad(F.cross_entropy(logits(adv),label),adv)[0]
            adv=(adv+0.005*grad.sign()).detach()
            adv=torch.max(torch.min(adv,pixel+0.03),pixel-0.03).clamp(0,1)
    return ((adv-mean)/std).detach()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',choices=['coverage','ood','adversarial','directional','linearity','trajectories','alignment'])
    p.add_argument('--config',default='configs/training/mnist_mlp.yaml')
    p.add_argument('--checkpoint',help='State dict produced by the training command; required for custom models')
    p.add_argument('--pretrained',choices=['resnet20','vgg16_bn','vgg16','resnet50','vit_base','deit_small'])
    p.add_argument('--data-path',default=None)
    p.add_argument('--ood-data-path',default='data/CIFAR10')
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto')
    p.add_argument('--k',type=int,default=0,help='Samples per true class; 0 selects all')
    p.add_argument('--pool-size',type=int,default=0,help='Training anchors per predicted class; 0 selects all')
    p.add_argument('--batch-size',type=int,default=100)
    p.add_argument('--step-size',type=float,default=0.01)
    p.add_argument('--max-eps',type=float,default=3.0)
    p.add_argument('--save-pairs',action='store_true')
    p.add_argument('--directions',type=int,default=784)
    p.add_argument('--scale',type=float,default=None)
    p.add_argument('--cosine-threshold',type=float,default=0.95)
    p.add_argument('--max-alpha',type=float,default=20.)
    p.add_argument('--alpha-steps',type=int,default=100)
    p.add_argument('--attack',choices=['cw','ifgsm','pgd'],default='pgd')
    p.add_argument('--attack-steps',type=int,default=None)
    args=p.parse_args(argv)
    if not args.pretrained and not args.checkpoint:p.error('--checkpoint is required for custom models')
    if args.pretrained and not args.data_path:p.error('--data-path is required for pretrained models')
    if args.k<0 or args.pool_size<0 or args.step_size<=0 or args.batch_size<=0 or args.max_eps<=1:p.error('Use nonnegative sample counts, positive step/batch sizes, and max-eps > 1')
    if args.experiment in ('coverage','ood','adversarial'):run_coverage(args,ood=args.experiment=='ood',attacks=args.experiment=='adversarial')
    elif args.experiment=='trajectories':run_trajectories(args)
    else:run_directional(args,args.experiment)

if __name__=='__main__':main()

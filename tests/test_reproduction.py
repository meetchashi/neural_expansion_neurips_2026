"""Offline scientific and integration checks; no data or weight downloads."""
import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import TensorDataset, Subset
from src.analysis.gi_calculation import compute_GI
from src.datasets import Relabeled, apply_training_variant, LT_COUNTS
from src.models.mlp import PlainNN
from src.models.cnn import CNN
from src.models.transformer import ViT
from src.reproduce import main, attack, jacobian_svd, feature_function

class LinearClassifier(torch.nn.Module):
    def forward(self,x):
        s=x.flatten(1).sum(1)
        return torch.stack([-s,s],1)
    def extract_features(self,x):
        return [x,x.flatten(1),self(x)]

class ReproductionTests(unittest.TestCase):
    def test_gi_first_boundary_and_anchor_class(self):
        m=LinearClassifier();x=torch.full((1,1,2,2),-1.);target=-x
        pos,neg=compute_GI(m,x,target,False,0,step_size=.01,max_eps=3.)
        self.assertAlmostEqual(pos[0],.51,places=5)
        self.assertEqual(neg[0],3.)
        pos,_=compute_GI(m,x,target,False,1,step_size=.01)
        self.assertEqual(pos[0],0.)

    def test_nullspace_and_singular_spectrum(self):
        m=PlainNN(input_dim=4,hidden_dim1=4,hidden_dim2=2)
        _,j,s,vh=jacobian_svd(m,torch.ones(1,2,2))
        self.assertEqual(vh.shape,(4,4))
        self.assertTrue(torch.allclose(vh@vh.T,torch.eye(4),atol=1e-5))
        self.assertTrue(torch.allclose(j@vh[2:].T,torch.zeros(2,2),atol=1e-5))

    def test_original_architectures_forward_and_features(self):
        for m in [PlainNN(hidden_dim1=1024,hidden_dim2=512),CNN(),ViT(depth=1)]:
            m.eval();x=torch.randn(2,1,28,28)
            self.assertEqual(m(x).shape,(2,10))
            self.assertEqual(m.extract_features(x)[-2].shape[0],2)

    def test_pretrained_feature_hook_preserves_gradients(self):
        class HeadModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 2)
            def forward(self, x): return self.fc(x.flatten(1))
        m=HeadModel().eval();x=torch.randn(1,1,2,2,requires_grad=True)
        features=feature_function(m)(x)
        self.assertTrue(torch.allclose(m.fc(features),m(x)))
        self.assertEqual(torch.autograd.grad(features.sum(),x)[0].shape,x.shape)
        self.assertEqual(len(m.fc._forward_pre_hooks),0)

    def test_random_labels_are_stable(self):
        d=TensorDataset(torch.zeros(100,1,2,2),torch.zeros(100,dtype=torch.long))
        a,b=Relabeled(d,100,42),Relabeled(d,100,42)
        np.testing.assert_array_equal(a.targets,b.targets)
        self.assertTrue(all(0<=a[i][1]<100 for i in range(len(a))))

    def test_long_tail_exact_counts_and_global_indices(self):
        labels=torch.arange(10).repeat_interleave(5000)
        d=TensorDataset(torch.zeros(50000,1),labels);sub=Subset(d,list(range(len(d))))
        for variant,counts in LT_COUNTS.items():
            actual=apply_training_variant(sub,{'variant':variant,'variant_seed':42})
            self.assertEqual(np.bincount([actual[i][1] for i in range(len(actual))]).tolist(),counts)
            self.assertEqual(len(set(actual.indices)),len(actual))

    def test_iterative_attacks_stay_in_pixel_bounds(self):
        m=LinearClassifier();x=torch.full((1,2,2),.1)
        cfg={'dataset':{'transform':{'normalize':{'mean':[.5],'std':[.5]}}}}
        for method in ['ifgsm','pgd','cw']:
            adv=attack(m,x,1,cfg,argparse.Namespace(attack=method,attack_steps=2))
            self.assertTrue(torch.isfinite(adv).all())
            self.assertLessEqual(float(adv.max()),1.)
            self.assertGreaterEqual(float(adv.min()),-1.)
            if method!='cw':self.assertLessEqual(float(((adv-x)*.5).abs().max()),.030001)

    def test_portable_runners_write_measured_outputs(self):
        train=TensorDataset(torch.stack([torch.full((1,2,2),-1.),torch.ones(1,2,2)]),torch.tensor([0,1]))
        cfg={'dataset':{'name':'MNIST','transform':{'normalize':{'mean':[.5],'std':[.5]}}}}
        for mode in ['coverage','directional','linearity','trajectories','alignment','adversarial']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                with patch('src.reproduce.load_run',return_value=(LinearClassifier(),cfg,train,train,torch.device('cpu'))):
                    main([mode,'--checkpoint','unused','--output',temp,'--k','1','--pool-size','1',
                          '--directions','2','--step-size','.25','--alpha-steps','3','--attack-steps','2'])
                expected='coverage.csv' if mode in ('coverage','adversarial') else 'trajectories.csv' if mode=='trajectories' else 'measurements.csv'
                self.assertTrue((Path(temp)/expected).exists())
                self.assertGreater(len(pd.read_csv(Path(temp)/expected)),0)

    def test_training_saves_reloadable_checkpoint_and_split(self):
        from src.training.mnist_train import train_model
        from src.runtime import load_custom
        d=TensorDataset(torch.randn(8,1,2,2),torch.arange(8)%2)
        with tempfile.TemporaryDirectory() as temp:
            cfg={'seed':42,'device':'cpu','dataset':{'name':'MNIST','val_split':.25,'num_classes':2},
                 'model':{'name':'PlainNN','file':'mlp','args':{'input_dim':4,'hidden_dim1':4,'hidden_dim2':2,'output_dim':2}},
                 'optimizer':{'name':'SGD','args':{'lr':.001}},'training':{'checkpoint_dir':temp,'batch_size':3,'num_epochs':1,'save_every':100}}
            path=Path(temp)/'input.yaml';path.write_text(yaml.safe_dump(cfg))
            with patch('src.training.mnist_train.get_dataset',return_value=(d,d)):train_model(path)
            self.assertTrue((Path(temp)/'model_epoch0_batch1.pth').exists())
            self.assertTrue((Path(temp)/'split_indices.npz').exists())
            model,_=load_custom(path,Path(temp)/'best_model_mnist.pth','cpu')
            self.assertEqual(model(torch.ones(1,1,2,2)).shape,(1,2))

if __name__=='__main__':unittest.main()

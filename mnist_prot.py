import argparse
import os
import uuid

import torch, torchvision, torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
import pytorch_lightning as pl

import gw

device = 'cuda'
g = gw.layers.GW2D(device=device)

tmp_dir = 'tmp'
class SelfDeletingTempFile():
    def __init__(self):
        self.name = os.path.join(tmp_dir, str(uuid.uuid4()))

    def __del__(self):
        os.remove(self.name)

def pack_hook(tensor):
    temp_file = SelfDeletingTempFile()
    torch.save(tensor, temp_file.name)
    return temp_file

def unpack_hook(temp_file):
    return torch.load(temp_file.name)

def gw2d_algebra(w:torch.Tensor) -> torch.Tensor:
    m = w.size()
    w_padded = torch.nn.functional.pad(w,[m[-2]//2,m[-2]//2,m[-1]//2,m[-1]//2])
    n = w_padded.stride()
    w_strided = torch.as_strided(
        w_padded,
        size=m[:-3]+m[-3:]+m[-2:],
        stride=n[:-3]+n[-3:]+n[-2:],
    ).flip([-4,-3])
    return -1j*torch.add(
        w_strided[...,0,:,:,:,:]*torch.arange(-m[-2]//2+1,m[-2]//2+1,device=w.device)[:,None],
        w_strided[...,1,:,:,:,:]*torch.arange(-m[-1]//2+1,m[-1]//2+1,device=w.device)[None,:]
    )
    
def gw2d_transform(w:torch.Tensor) -> torch.Tensor:
    size = w.size()[-2:]
    return torch.matrix_exp(
        gw2d_algebra(w).flatten(-4,-3).flatten(-2,-1)
    ).unflatten(-2,size).unflatten(-1,size)
    
def transform_to_map(transform:torch.Tensor,input_size:int,output_size:int) -> torch.Tensor:
    nt = transform.size()[-4:]
    ni = input_size
    no = output_size
    di = nt[2]-ni[0],nt[3]-ni[1]
    do = nt[0]-no[0],nt[1]-no[1]
    x = transform
    x = x[...,do[0]//2+1:-do[0]//2+1,do[1]//2+1:-do[1]//2+1,di[0]//2+1:-di[0]//2+1,di[1]//2+1:-di[1]//2+1]
    x = x.roll((no[0]//2+1,no[1]//2+1,ni[0]//2+1,ni[1]//2+1),(-4,-3,-2,-1))
    map = torch.fft.fft2(torch.fft.ifft2(x,dim=(-2,-1)),dim=(-4,-3)).real
    return map

def gw2d_map(w:torch.Tensor,input_size:int,output_size:int) -> torch.Tensor:
    return transform_to_map(gw2d_transform(w),input_size,output_size)

def signal_to_spectrum(signal:torch.Tensor):
    n = signal.size()[-2:]
    window = torch.hann_window(n[0],device=signal.device)[:,None]*torch.hann_window(n[1],device=signal.device)[None,:]
    rx = torch.arange(-n[0]//2+1,n[0]//2+1,device=signal.device)[:,None]
    ry = torch.arange(-n[1]//2+1,n[1]//2+1,device=signal.device)[None,:]
    r = 1+rx**2+ry**2
    wt = signal
    wf = torch.fft.fft2(wt).roll((n[0]//2,n[1]//2),(-2,-1))/r/5
    wt = torch.fft.ifft2(wf.roll((-(n[0]//2),-(n[1]//2)),(-2,-1)))*window
    spectrum = torch.fft.fft2(wt).roll((n[0]//2,n[1]//2),(-2,-1))
    return spectrum

class GWPooling2D(torch.nn.Module):
    def __init__(self, input_features, output_features, output_channels, parameter_size) -> None:
        super().__init__()
        self.signal = torch.nn.Parameter(torch.zeros((output_channels,2)+parameter_size))
        self.input_size = input_features
        self.output_size = output_features
        self.reset_parameters()
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        spectrum = signal_to_spectrum(self.signal)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            transform = gw2d_transform(spectrum)
            map = transform_to_map(transform, self.input_size, self.output_size)
            y = (map.flatten(-4,-3).flatten(-2,-1)@x.flatten(-2).unsqueeze(-1)).squeeze(-1).unflatten(-1,self.output_size)
        return y
    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.signal)

class GW_CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = self.create_model()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.train_acc = torchmetrics.Accuracy('multiclass', num_classes=10)
        self.val_acc = torchmetrics.Accuracy('multiclass', num_classes=10)
        self.test_acc = torchmetrics.Accuracy('multiclass', num_classes=10)
        # self.save_hyperparameters({**model, **optimizer}, logger=False)
      
    def create_model(self):
        model = torch.nn.Sequential(
            GWPooling2D((28,28),(28,28),4,(33,33)),
            torch.nn.Conv2d(4,4,3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4,4,3),
            torch.nn.ReLU(),
            GWPooling2D((24,24),(12,12),4,(33,33)),
            torch.nn.Conv2d(4,4,3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4,4,3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(8*8*4,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,10)
        )
        # モデル構成の表示
        print(model)
        return model
      
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log('train/loss', loss, on_epoch=False, on_step=True, prog_bar=False, logger=True)
        self.log('train/acc', self.train_acc(pred,y), on_epoch=False, on_step=True, prog_bar=True, logger=True)
        if batch_idx%1==0:
            self.save_warps(self.trainer.global_step)
        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        acc = self.val_acc(pred,y)
        self.log('val/acc', acc, prog_bar=True, logger=True)
        return acc
    
    def save_warps(self, step) -> None:
        t = torch.linspace(0,4*torch.pi,28)
        x = torch.sin(t)[:,None]
        y = torch.sin(t)[None,:]
        z = x+y
        with torch.no_grad():
            z_ = self.model[0](z[None].cuda())
        fig,ax = plt.subplots(2,2,figsize=(8,8))
        ax[0,0].imshow(z_[0].cpu())
        ax[0,1].imshow(z_[1].cpu())
        ax[1,0].imshow(z_[2].cpu())
        ax[1,1].imshow(z_[3].cpu())
        plt.close(fig)
        self.logger.experiment.add_figure("Warp_0", fig, step)
        
        t = torch.linspace(0,4*torch.pi,24)
        x = torch.sin(t)[:,None]
        y = torch.sin(t)[None,:]
        z = x+y
        plt.imshow(z)
        plt.show()
        with torch.no_grad():
            z_ = self.model[5](z[None].cuda())
        fig,ax = plt.subplots(2,2,figsize=(8,8))
        ax[0,0].imshow(z_[0].cpu())
        ax[0,1].imshow(z_[1].cpu())
        ax[1,0].imshow(z_[2].cpu())
        ax[1,1].imshow(z_[3].cpu())
        self.logger.experiment.add_figure("Warp_1", fig, step)
    
    def test_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        acc = self.test_acc(pred, y)
        self.log('test/acc', acc, prog_bar=True, logger=True)
        return acc
    
    def configure_optimizers(self):
        return self.optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('transform',type=str,default='none')
    args = parser.parse_args()
    
    if args.transform == 'rotation':
        print('transform: rotation')
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomAffine(degrees=180)
            ]
        )
    elif args.transform == 'translation':
        print('transform: translation')
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomAffine(degrees=0,translate=(0.5,0.5))
            ]
        )
    elif args.transform == 'scaling':
        print('transform: scaling')
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomAffine(degrees=0,scale=(0.5,2))
            ]
        )
    else:
        print('transform: none')
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor()
            ]
        )
    train_dataset = torchvision.datasets.MNIST('./data',train=True,download=True,transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data',train=False,download=True,transform=transform)
    
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        test_dataset=test_dataset,
        batch_size=256,
        num_workers=4)
    
    model = GW_CNN()
    logger = TensorBoardLogger('exp',args.transform)
    trainer = pl.Trainer(max_epochs=5, accelerator='gpu',logger=logger)
    
    # モデルの学習
    trainer.fit(model=model, datamodule=datamodule)
    
    # バリデーション
    trainer.validate(model=model, datamodule=datamodule)
    
    # テスト
    trainer.test(model=model, datamodule=datamodule)
    

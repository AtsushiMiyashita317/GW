import argparse
import os
import uuid

import torch, torchvision, torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
import pytorch_lightning as pl

import gw

class GW2D:
    def __arange(self, size:torch.Size, device=None):
        x = torch.arange(-size[0]//2+1,size[0]//2+1,device=device)
        y = torch.arange(-size[1]//2+1,size[1]//2+1,device=device)
        return x[:,None], y[None,:]
    
    def __pad(self, w:torch.Tensor, scale:int, value=None):
        px = (w.size(-2)//2)*(scale-1)
        py = (w.size(-1)//2)*(scale-1)
        return torch.nn.functional.pad(w,[py,py,px,px],value=value)
    
    def __as_strided(self, w:torch.Tensor):
        size = (w.size(-2)//2+1,w.size(-1)//2+1)
        return torch.as_strided(
            w,
            size=w.size()[:-2]+size+size,
            stride=w.stride()[:-2]+w.stride()[-2:]+w.stride()[-2:]
        ).flip([-2,-1])
    
    def algebra(self, w:torch.Tensor):
        L = self.__as_strided(self.__pad(w,2))
        x,y = self.__arange(L.size()[-2:],device=w.device)
        return -1j*(L[...,0,:,:,:,:]*x+L[...,1,:,:,:,:]*y).flatten(-4,-3).flatten(-2,-1)
        
    def element(self, w: torch.Tensor) -> torch.Tensor:
        return torch.matrix_exp(self.algebra(w))
            
    def signal_to_spectrum(self, signal:torch.Tensor) -> torch.Tensor:
        x,y = self.__arange((signal.size(-2)+1,signal.size(-1)+1))
        return torch.fft.fft2(signal,dim=[-2,-1])[...,x,y]
    
    def transform_to_map(self, transform:torch.Tensor, size:torch.Tensor) -> torch.Tensor:
        n = size[-2]//2,size[-1]//2
        return torch.fft.ifft2(
            torch.fft.fft2(
                transform.unflatten(-2,size).unflatten(-1,size)[...,:-1,:-1,:-1,:-1].roll((n[0],n[1],n[0],n[1]),(-4,-3,-2,-1)),
                dim=[-2,-1]
            ),
            dim=[-4,-3]
        ).real
    
    def map(self, signal:torch.Tensor) -> torch.Tensor:
        """Create GW map from GW signal

        Args:
            signal (torch.Tensor): GW signal

        Returns:
            torch.Tensor: GW map
            
        Notes:
            signal length must be even
        """
        return self.transform_to_map(self.element(self.signal_to_spectrum(signal)),(signal.size(-2)+1,signal.size(-1)+1))
    
    def warp(self, signal:torch.Tensor, input:torch.Tensor) -> torch.Tensor:
        map = self.map(signal)
        return map.flatten(-4,-3).flatten(-2,-1).matmul(input.flatten(-2,-1).unsqueeze(-1)).squeeze(-1).unflatten(-1,input.size()[-2:])

class stgw_impl(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx:torch.autograd.function.FunctionCtx, 
            x:torch.Tensor, 
            map:torch.Tensor, 
            map_inv:torch.Tensor,
            n_iter:int
        ):
        n = map.size()[-2:]
        h = n[0]//2, n[1]//2
        rsize = x.size()[:-2]+(map.size(-9),)+x.size()[-2:]
        a = x.new_zeros(rsize)
        b = x.new_zeros(rsize)
        a.copy_(x.unsqueeze(-3))
        size = a.size()[:-2] + (2, 2, a.size(-2)//n[0], a.size(-1)//n[1], n[0], n[1])
        stride = a.stride()[:-2] + (h[0]*a.stride(-2), h[1]*a.stride(-1), n[0]*a.stride(-2), n[1]*a.stride(-1), a.stride(-2), a.stride(-1))
        a_strided = a.as_strided(size, stride)
        b_strided = b.as_strided(size, stride)
        
        for _ in range(n_iter):
            b.copy_(a)
            b_strided[...,0,0,:,:,:,:] = torch.einsum('...nabijkl,...mnabkl->...mnabij', map[...,0,0,:,:,:,:,:,:], a_strided[...,0,0,:,:,:,:])
            a.copy_(b)
            a_strided[...,0,1,:,:,:,:] = torch.einsum('...nabijkl,...mnabkl->...mnabij', map[...,0,1,:,:,:,:,:,:], b_strided[...,0,1,:,:,:,:])
            b.copy_(a)
            b_strided[...,1,1,:,:,:,:] = torch.einsum('...nabijkl,...mnabkl->...mnabij', map[...,1,1,:,:,:,:,:,:], a_strided[...,1,1,:,:,:,:])
            a.copy_(b)
            a_strided[...,1,0,:,:,:,:] = torch.einsum('...nabijkl,...mnabkl->...mnabij', map[...,1,0,:,:,:,:,:,:], b_strided[...,1,0,:,:,:,:])
        y = a
        ctx.save_for_backward(y, map, map_inv)
        ctx.n_iter = n_iter
        return y
    
    @staticmethod
    def backward(ctx:torch.autograd.function.FunctionCtx, dy:torch.Tensor):
        y:torch.Tensor
        map:torch.Tensor
        map_inv:torch.Tensor
        y, map, map_inv = ctx.saved_tensors
        n_iter = ctx.n_iter
        dmap = map.new_zeros(map.size())
        
        n = map.size()[-2:]
        h = n[0]//2, n[1]//2
        
        a = y.new_zeros(y.size())
        b = y.new_zeros(y.size())
        da = dy.new_zeros(dy.size())
        db = dy.new_zeros(dy.size())
        
        a.copy_(y)
        da.copy_(dy)
        
        size = a.size()[:-2] + (2, 2, a.size(-2)//n[0], a.size(-1)//n[1], n[0], n[1])
        stride = a.stride()[:-2] + (h[0]*a.stride(-2), h[1]*a.stride(-1), n[0]*a.stride(-2), n[1]*a.stride(-1), a.stride(-2), a.stride(-1))
        
        a_strided = a.as_strided(size, stride)
        b_strided = b.as_strided(size, stride)
        da_strided = da.as_strided(size, stride)
        db_strided = db.as_strided(size, stride)
        
        for _ in range(n_iter):
            b.copy_(a)
            b_strided[...,1,0,:,:,:,:] = torch.einsum('...nabijkl,...mnabkl->...mnabij', map_inv[...,1,0,:,:,:,:,:,:], a_strided[...,1,0,:,:,:,:])
            dmap[...,1,0,:,:,:,:,:,:] += torch.einsum('...mnabij,...mnabkl->...nabijkl', da_strided[...,1,0,:,:,:,:], b_strided[...,1,0,:,:,:,:])
            db.copy_(da)
            db_strided[...,1,0,:,:,:,:] = torch.einsum('...nabijkl,...mnabij->...mnabkl', map[...,1,0,:,:,:,:,:,:], da_strided[...,1,0,:,:,:,:])
            a.copy_(b)
            a_strided[...,1,1,:,:,:,:] = torch.einsum('...nabijkl,...mnabkl->...mnabij', map_inv[...,1,1,:,:,:,:,:,:], b_strided[...,1,1,:,:,:,:])
            dmap[...,1,1,:,:,:,:,:,:] += torch.einsum('...mnabij,...mnabkl->...nabijkl', db_strided[...,1,1,:,:,:,:], a_strided[...,1,1,:,:,:,:])
            da.copy_(db)
            da_strided[...,1,1,:,:,:,:] = torch.einsum('...nabijkl,...mnabij->...mnabkl', map[...,1,1,:,:,:,:,:,:], db_strided[...,1,1,:,:,:,:])
            b.copy_(a)
            b_strided[...,0,1,:,:,:,:] = torch.einsum('...nabijkl,...mnabkl->...mnabij', map_inv[...,0,1,:,:,:,:,:,:], a_strided[...,0,1,:,:,:,:])
            dmap[...,0,1,:,:,:,:,:,:] += torch.einsum('...mnabij,...mnabkl->...nabijkl', da_strided[...,0,1,:,:,:,:], b_strided[...,0,1,:,:,:,:])
            db.copy_(da)
            db_strided[...,0,1,:,:,:,:] = torch.einsum('...nabijkl,...mnabij->...mnabkl', map[...,0,1,:,:,:,:,:,:], da_strided[...,0,1,:,:,:,:])
            a.copy_(b)
            a_strided[...,0,0,:,:,:,:] = torch.einsum('...nabijkl,...mnabkl->...mnabij', map_inv[...,0,0,:,:,:,:,:,:], b_strided[...,0,0,:,:,:,:])
            dmap[...,0,0,:,:,:,:,:,:] += torch.einsum('...mnabij,...mnabkl->...nabijkl', db_strided[...,0,0,:,:,:,:], a_strided[...,0,0,:,:,:,:])
            da.copy_(db)
            da_strided[...,0,0,:,:,:,:] = torch.einsum('...nabijkl,...mnabij->...mnabkl', map[...,0,0,:,:,:,:,:,:], db_strided[...,0,0,:,:,:,:])
        
        dx = da.sum(-3)
        return dx, dmap, None, None

g = GW2D()

def lpf(signal:torch.Tensor):
    c0 = torch.arange(-signal.size(-2)+1, signal.size(-2)+1,device=signal.device).roll(signal.size(-2)+1, -1).abs()
    c1 = torch.arange(signal.size(-1)+1,device=signal.device)
    c0[0] = 1
    c1[0] = 1
    return torch.fft.irfft2(
        torch.fft.rfft2(
            torch.nn.functional.pad(signal,[0,signal.size(-1),0,signal.size(-2)]),
        ).div(c0.unsqueeze(-1)**2*c1.unsqueeze(-2)**2)
    )[...,:signal.size(-2),:signal.size(-1)]

def padding_for_stgw(gw_signal:torch.Tensor,signal:torch.Tensor,window_size:torch.Size):
    # calculate size
    k = window_size
    l = signal.size()[-2:]
    h = k[0]//2,k[1]//2
    n = (l[0]+h[0]-1)//h[0],(l[1]+h[1]-1)//h[1]
    n = n[0]//2+1,n[1]//2+1
    m = k[0]*n[0]+h[0],k[1]*n[1]+h[1]
    # padding
    p = h[1],m[1]-l[1]-h[1],h[0],m[0]-l[0]-h[0]
    signal = torch.nn.functional.pad(signal,p)
    gw_signal = torch.nn.functional.pad(gw_signal,p)
    return gw_signal,signal,l

def strip_for_stgw(signal:torch.Tensor,length:torch.Size,window_size:torch.Size):
    k = window_size
    l = length
    return signal[...,k[0]//2:k[0]//2+l[0],k[1]//2:k[1]//2+l[1]]


def stgw_aug(gw_signal:torch.Tensor, input:torch.Tensor, window_size:torch.Size, n_iter:int, k_iter:int) -> torch.Tensor:
    k = window_size
    h = k[0]//2,k[1]//2
    w = lpf(gw_signal)
    w,x,l = padding_for_stgw(w, input, k)
    n = w.size()[-2:]
    
    size = w.size()[:-3] + (2, 2, n[0]//k[0], n[1]//k[1], 2, k[0], k[1])
    stride = w.stride()[:-3] + (h[0]*w.stride(-2), h[1]*w.stride(-1), k[0]*w.stride(-2), k[1]*w.stride(-1), w.stride(-3), w.stride(-2), w.stride(-1))
    w = w.as_strided(
        size, stride
    ).mul(
        torch.hann_window(k[0], dtype=w.dtype, device=w.device).unsqueeze(-1)
    ).mul(
        torch.hann_window(k[1], dtype=w.dtype, device=w.device).unsqueeze(-2)
    )
    w[...,0,:,:] = w[...,0,:,:]*(n[0]/k[0])**2*(n[1]/k[1])
    w[...,1,:,:] = w[...,1,:,:]*(n[1]/k[1])**2*(n[0]/k[0])
    w = w/k_iter/n_iter*5e-3
    map = g.map(w)
    map_inv = g.map(-w)
    
    y = x.new_zeros((k_iter,)+x.size())
    y[k_iter//2] = x
    for i in range(k_iter//2):
        y[k_iter//2+i+1] = stgw_impl.apply(y[k_iter//2+i], map, map_inv, n_iter).squeeze(2)
    for i in range(k_iter//2):
        y[k_iter//2-i-1] = stgw_impl.apply(y[k_iter//2-i], map_inv, map, n_iter).squeeze(2)
    
    output = strip_for_stgw(y, l, k)
    
    return output

class GWAug2D(torch.nn.Module):
    def __init__(self, input_features, k_iter) -> None:
        super().__init__()
        self.signal = torch.nn.Parameter(torch.zeros((1,1,2)+input_features))
        self.k_iter = k_iter
        self.reset_parameters()
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x.transpose(0,1)
        y = stgw_aug(self.signal, x, (4,4), 64, self.k_iter).transpose(0,2)
        return y
    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.signal)
        
def color_circle(x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
    z = - x + 1j*y
    C = z.abs().mul(10).tanh()
    H = (z.angle()+torch.pi)/(torch.pi/3)
    X = C*(1-torch.abs(H%2-1))
    O = torch.zeros_like(C)
    rgb = O.new_zeros(O.size()+(3,))
    rgb[:] = 1-C[...,None]
    rgb[(0<=H)&(H<1)] += torch.stack([C,X,O],dim=-1)[(0<=H)&(H<1)]
    rgb[(1<=H)&(H<2)] += torch.stack([X,C,O],dim=-1)[(1<=H)&(H<2)]
    rgb[(2<=H)&(H<3)] += torch.stack([O,C,X],dim=-1)[(2<=H)&(H<3)]
    rgb[(3<=H)&(H<4)] += torch.stack([O,X,C],dim=-1)[(3<=H)&(H<4)]
    rgb[(4<=H)&(H<5)] += torch.stack([X,O,C],dim=-1)[(4<=H)&(H<5)]
    rgb[(5<=H)&(H<6)] += torch.stack([C,O,X],dim=-1)[(5<=H)&(H<6)]
    return rgb

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
            GWAug2D((28,28),9),
            torch.nn.Conv3d(1,4,3),
            torch.nn.ReLU(),
            torch.nn.Conv3d(4,4,3),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d((1,2,2)),
            torch.nn.Conv3d(4,4,3),
            torch.nn.ReLU(),
            torch.nn.Conv3d(4,4,3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(8*8*4,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,10)
        )
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
        # t = torch.linspace(0,4*torch.pi,28)
        # z = t.cos().unsqueeze(-1)*t.cos()
        # with torch.no_grad():
        #     z_ = self.model[0](z[None,None].cuda())
        w = lpf(self.model[0].signal)
        flow = color_circle(w[0,0,1], w[0,0,0]).permute([2,0,1])
        self.logger.log_image("Warp_0", [flow], step)
    
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
                torchvision.transforms.RandomAffine(degrees=30)
            ]
        )
    elif args.transform == 'translation':
        print('transform: translation')
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomAffine(degrees=0,translate=(0.2,0.2))
            ]
        )
    elif args.transform == 'scaling':
        print('transform: scaling')
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomAffine(degrees=0,scale=(0.5,1.5))
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
    # logger = TensorBoardLogger('exp/gw-cnn',args.transform)
    logger = WandbLogger(name=args.transform, project='gwcnn-mnist',save_dir='exp', notes='test', save_code=True)
    trainer = pl.Trainer(max_epochs=5, accelerator='gpu',logger=logger)
    
    # モデルの学習
    trainer.fit(model=model, datamodule=datamodule)
    
    # バリデーション
    trainer.validate(model=model, datamodule=datamodule)
    
    # テスト
    trainer.test(model=model, datamodule=datamodule)
    

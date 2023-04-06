import torch
from torch.nn.functional import pad
import mytorch

class rgw:
    @staticmethod
    def signal_to_spectrum(signal:torch.Tensor)->torch.Tensor:
        spectrum = mytorch.fft.dst(signal)
        return spectrum

    @staticmethod
    def spectrum_to_signal(spectrum:torch.Tensor)->torch.Tensor:
        signal = mytorch.fft.dst(spectrum)
        return signal

    @staticmethod
    def spectrum_to_operator(spectrum:torch.Tensor,norm=False)->torch.Tensor:
        w = spectrum
        dim = w.shape[-1]
        if norm:
            w = w/torch.arange(1,dim,device=w.device)
        w = pad(torch.cat([-w.flip([-1]),torch.zeros_like(w[...,:1]),w],dim=-1),(dim-3,dim-3))
        L = torch.as_strided(
                w,
                size=w.shape[:-1]+((dim-1)*2,(dim-1)*2),
                stride=w.stride()[:-1]+(w.stride(-1),w.stride(-1))
            ).flip([-1]).mul(
                torch.arange(-dim+2,dim,device=w.device)
            )
        operator = mytorch.sp.biasing(L,dim=[-2,-1])
        return operator

    @staticmethod
    def operator_to_transform(operator:torch.Tensor)->torch.Tensor:
        transform = torch.matrix_exp(operator)
        return transform

    @staticmethod
    def transform_to_ir(transform:torch.Tensor)->torch.Tensor:
        ir = transform[...,1]
        return ir

    @staticmethod
    def ir_to_filter(ir:torch.Tensor)->torch.Tensor:
        filter = torch.fft.rfft(ir)
        return filter

    @staticmethod
    def filter_to_ir(filter:torch.Tensor)->torch.Tensor:
        ir = torch.fft.irfft(filter)
        return ir

    @staticmethod
    def filter_to_transform(filter:torch.Tensor)->torch.Tensor:
        size = filter.size(-1)
        deg =   mytorch.sp.biasing(
                    torch.arange(-size+2,size)
                )
        transform = torch.fft.irfft(
                        mytorch.functional.normalize(
                            torch.pow(
                                mytorch.functional.normalize(filter)[...,None],
                                deg
                            )
                        ),
                        dim=-2
                    )
        return transform
        
    @staticmethod
    def function_to_filter(function:torch.Tensor)->torch.Tensor:
        filter = torch.exp(-1j*torch.pi*function)
        return filter

    @staticmethod
    def filter_to_function(filter:torch.Tensor)->torch.Tensor:
        function = -filter.angle()/torch.pi
        function[function<0] += 2
        return function

    @staticmethod
    def transform_to_map(transform:torch.Tensor)->torch.Tensor:
        size = transform.size(-1)
        map = torch.fft.ifft(torch.fft.fft(transform,dim=-2),dim=-1)[...,:size//2+1,:size//2+1].real
        return map

    @staticmethod
    def map_to_transform(map:torch.Tensor)->torch.Tensor:
        size = map.size(-1)
        map =   torch.cat(
                    [
                        torch.cat([map,torch.zeros(size,size-2)],dim=-1),
                        torch.cat([torch.zeros(size-2,size),map[...,1:-1,1:-1].flip([-2,-1])],dim=-1)
                    ],
                    dim=-2
                )
        transform = torch.fft.ifft(
            torch.fft.fft(
                map,
                dim=-1
            ),
            dim=-2
        )
        return transform

    @staticmethod
    def warp_signal(map:torch.Tensor, signal:torch.Tensor)->torch.Tensor:
        return (map@signal[...,None])[...,0]

    @staticmethod
    def warp_spectrum(transform:torch.Tensor, spectrum:torch.Tensor)->torch.Tensor:
        return (transform@spectrum[...,None])[...,0]


class gw:
    @staticmethod
    def signal_to_spectrum(signal:torch.Tensor)->torch.Tensor:
        spectrum = torch.fft.rfft(signal)
        return spectrum

    @staticmethod
    def spectrum_to_signal(spectrum:torch.Tensor)->torch.Tensor:
        signal = torch.fft.irfft(spectrum)
        return signal

    @staticmethod
    def spectrum_to_operator(spectrum:torch.Tensor,norm=False)->torch.Tensor:
        w = spectrum
        dim = (w.shape[-1]-1)*2
        if norm:
            w[...,1:] = w[...,1:]/torch.arange(1,w.shape[-1],device=w.device)
        w = pad(mytorch.sp.centering(mytorch.sp.onside_to_bothside(w.conj(), even=False)),[w.shape[-1]-2,w.shape[-1]-2])
        L = torch.as_strided(
                w,
                size=w.shape[:-1]+(dim,dim),
                stride=w.stride()[:-1]+(w.stride(-1),w.stride(-1))
            ).flip([-1]).mul(
                1j*torch.arange(-dim//2+1,dim//2+1,device=w.device)
            )
        operator = mytorch.sp.biasing(L,dim=[-2,-1])
        return operator

    @staticmethod
    def operator_to_transform(operator:torch.Tensor)->torch.Tensor:
        # transform = torch.matrix_exp(operator)
        transform = torch.eye(operator.size(-1),device=operator.device,dtype=operator.dtype)+operator
        return transform

    @staticmethod
    def transform_to_ir(transform:torch.Tensor)->torch.Tensor:
        ir = transform[...,1]
        return ir

    @staticmethod
    def ir_to_filter(ir:torch.Tensor)->torch.Tensor:
        filter = torch.fft.fft(ir)
        return filter

    @staticmethod
    def filter_to_ir(filter:torch.Tensor)->torch.Tensor:
        ir = torch.fft.ifft(filter)
        return ir

    @staticmethod
    def filter_to_transform(filter:torch.Tensor)->torch.Tensor:
        size = filter.size(-1)
        deg =   mytorch.sp.biasing(
                    torch.arange(size)-(size-1)//2
                )
        transform = torch.fft.ifft(
                        mytorch.functional.normalize(
                            torch.pow(
                                mytorch.functional.normalize(filter)[...,None],
                                deg
                            )
                        ),
                        dim=-2
                    )
        return transform
        
    @staticmethod
    def function_to_filter(function:torch.Tensor)->torch.Tensor:
        filter = torch.exp(-2j*torch.pi*function)
        return filter

    @staticmethod
    def filter_to_function(filter:torch.Tensor)->torch.Tensor:
        function = -filter.angle()/(2*torch.pi)
        function[function<0] += 1
        return function

    @staticmethod
    def transform_to_map(transform:torch.Tensor)->torch.Tensor:
        map = torch.fft.ifft(torch.fft.fft(transform,dim=-2),dim=-1).real
        return map

    @staticmethod
    def map_to_transform(map:torch.Tensor, odd=False)->torch.Tensor:
        transform = torch.fft.ifft(torch.fft.fft(map,dim=-1),dim=-2)
        return transform

    @staticmethod
    def warp_signal(map:torch.Tensor, signal:torch.Tensor)->torch.Tensor:
        return (map@signal[...,None])[...,0]

    @staticmethod
    def warp_spectrum(transform:torch.Tensor, spectrum:torch.Tensor)->torch.Tensor:
        return (transform@spectrum[...,None])[...,0]
    
    @staticmethod
    def padding_for_stgw(gw_signal:torch.Tensor,signal:torch.Tensor,window_size:int=16):
        # calculate size
        length = signal.size(-1)
        hop_length = window_size//2
        n_window = (length+hop_length-1)//hop_length
        n_window = n_window//2+1
        whole_length = window_size*n_window+hop_length
        # padding
        pad = hop_length,whole_length-length-hop_length
        signal = torch.nn.functional.pad(signal,pad)
        gw_signal = torch.nn.functional.pad(gw_signal,pad)
        return gw_signal,signal,length
    
    @staticmethod
    def strip_for_stgw(signal:torch.Tensor,length:int,window_size:int=16):
        return signal[...,window_size//2:window_size//2+length]
    
    @staticmethod
    def short_time_warp(gw_signal:torch.Tensor, signal:torch.Tensor, window_size:int=16, n:int=16)->torch.Tensor:
        """short time warp with hanning window

        Args:
            gw_signal (torch.Tensor, axis=(...,time))
            signal (torch.Tensor, axis=(...,time))
            window_size (int)
            n (int): The iteration number for scaling-squareing

        Returns:
            torch.Tensor: output signal
        """
        # calculate size
        length = signal.size(-1)
        hop_length = window_size//2
        n_window = length//window_size
        # stiride GW signal
        s = gw_signal.stride(-1)
        # (...,2,n_window,window_size)
        gw_signal = torch.as_strided(
            gw_signal,
            size=gw_signal.size()[:-1]+(2,n_window,window_size),
            stride=gw_signal.stride()[:-1]+(hop_length*s,window_size*s,s)
        )
        # windowing
        gw_signal = gw_signal*torch.hann_window(window_size,device=gw_signal.device)/n
        # calculate GW map
        # (...,2,n_window,window_size,window_size)
        gw_map =    gw.transform_to_map(
                        gw.operator_to_transform(
                            gw.spectrum_to_operator(
                                gw.signal_to_spectrum(gw_signal))))
        # warp
        size = signal.size()[:-1]+(n_window,window_size)
        for _ in range(n):
            signal = torch.cat(
                [
                    gw.warp_signal(
                        gw_map[...,0,:,:,:],
                        signal[...,:-hop_length].reshape(size)
                    ).flatten(start_dim=-2),
                    signal[:,:,-hop_length:]
                ],
                dim=-1
            )
            signal = torch.cat(
                [
                    signal[:,:,:hop_length],
                    gw.warp_signal(
                        gw_map[...,1,:,:,:],
                        signal[...,hop_length:].reshape(size)
                    ).flatten(start_dim=-2),
                ],
                dim=-1
            )
        
        return signal


    @staticmethod
    def inverse_short_time_warp(gw_signal:torch.Tensor, signal:torch.Tensor, window_size:int=16, n:int=16)->torch.Tensor:
        """inverse short time warp with hanning window

        Args:
            gw_signal (torch.Tensor, axis=(...,time))
            signal (torch.Tensor, axis=(...,time))
            window_size (int)
            n (int): The iteration number for scaling-squareing

        Returns:
            torch.Tensor: output signal
        """
        # calculate size
        length = signal.size(-1)
        hop_length = window_size//2
        n_window = length//window_size
        # stiride GW signal
        s = gw_signal.stride(-1)
        # (...,2,n_window,window_size)
        gw_signal = torch.as_strided(
            gw_signal,
            size=gw_signal.size()[:-1]+(2,n_window,window_size),
            stride=gw_signal.stride()[:-1]+(hop_length*s,window_size*s,s)
        )
        # windowing
        gw_signal = -gw_signal*torch.hann_window(window_size,device=gw_signal.device)/n
        # calculate GW map
        # (...,2,n_window,window_size,window_size)
        gw_map =    gw.transform_to_map(
                        gw.operator_to_transform(
                            gw.spectrum_to_operator(
                                gw.signal_to_spectrum(gw_signal))))
        # warp
        size = signal.size()[:-1]+(n_window,window_size)
        for _ in range(n):
            signal = torch.cat(
                [
                    signal[...,:hop_length],
                    gw.warp_signal(
                        gw_map[...,1,:,:,:],
                        signal[...,hop_length:].reshape(size)
                    ).flatten(start_dim=-2),
                ],
                dim=-1
            )
            signal = torch.cat(
                [
                    gw.warp_signal(
                        gw_map[...,0,:,:,:],
                        signal[...,:-hop_length].reshape(size)
                    ).flatten(start_dim=-2),
                    signal[...,-hop_length:]
                ],
                dim=-1
            )
            
        return signal


class gw2:
    @staticmethod
    def signal_to_spectrum(signal:torch.Tensor)->torch.Tensor:
        spectrum = torch.fft.fft2(signal)
        return spectrum

    @staticmethod
    def spectrum_to_signal(spectrum:torch.Tensor)->torch.Tensor:
        signal = torch.fft.ifft2(spectrum)
        return signal

    @staticmethod
    def spectrum_to_operator(spectrum:torch.Tensor,norm=False)->torch.Tensor:
        w = spectrum
        dimx = w.size(-2)
        dimy = w.size(-1)
        if norm:
            # w[...,1:] = w[...,1:]/torch.arange(1,w.shape[-1],device=w.device)
            pass
        w = pad(mytorch.sp.centering(w,[-2,-1]),[dimy//2,dimy//2-1,dimx//2,dimx//2-1])
        L = torch.as_strided(
                w,
                size=w.size()[:-2]+(dimx,dimy,dimx,dimy),
                stride=w.stride()+(w.stride(-2),w.stride(-1))
            ).flip([-4,-3]).mul(1j)
        operator = mytorch.sp.biasing(
            L[...,0,:,:,:,:].mul(torch.arange(-dimx//2+1,dimx//2+1,device=w.device)[:,None])+\
            L[...,1,:,:,:,:].mul(torch.arange(-dimy//2+1,dimy//2+1,device=w.device)[None,:]),
            dim=[-4,-3,-2,-1]
        )
        return operator

    @staticmethod
    def operator_to_transform(operator:torch.Tensor)->torch.Tensor:
        size2 = operator.size()
        size1 = size2[:-4] + (size2[-4]*size2[-3],size2[-2]*size2[-1])
        transform = torch.matrix_exp(operator.reshape(size1)).reshape(size2)
        return transform

    @staticmethod
    def transform_to_ir(transform:torch.Tensor)->torch.Tensor:
        ir = torch.stack([transform[...,1,0],transform[...,0,1]],dim=-3)
        return ir

    @staticmethod
    def ir_to_filter(ir:torch.Tensor)->torch.Tensor:
        filter = torch.fft.fft2(ir)
        return filter

    @staticmethod
    def filter_to_ir(filter:torch.Tensor)->torch.Tensor:
        ir = torch.fft.ifft2(filter)
        return ir

    @staticmethod
    def filter_to_transform(filter:torch.Tensor)->torch.Tensor:
        size = filter.size(-1)
        deg =   mytorch.sp.biasing(
                    torch.arange(size)-(size-1)//2
                )
        transform = torch.fft.ifft2(
                        mytorch.functional.normalize(
                            torch.pow(
                                mytorch.functional.normalize(filter)[...,None,None],
                                deg[None,:]
                            )*\
                            torch.pow(
                                mytorch.functional.normalize(filter)[...,None,None],
                                deg[:,None]
                            )   
                        )
                    )
        return transform
        
    @staticmethod
    def function_to_filter(function:torch.Tensor)->torch.Tensor:
        filter = torch.exp(-2j*torch.pi*function)
        return filter

    @staticmethod
    def filter_to_function(filter:torch.Tensor)->torch.Tensor:
        function = -filter.angle()/(2*torch.pi)
        function[function<0] += 1
        return function

    @staticmethod
    def transform_to_map(transform:torch.Tensor)->torch.Tensor:
        map = torch.fft.ifft2(torch.fft.fft2(transform,dim=(-4,-3)),dim=(-2,-1)).real
        return map

    @staticmethod
    def map_to_transform(map:torch.Tensor, odd=False)->torch.Tensor:
        transform = torch.fft.ifft2(torch.fft.fft2(map,dim=(-2,-1)),dim=(-4,-3))
        return transform

    @staticmethod
    def warp_signal(map:torch.Tensor, signal:torch.Tensor)->torch.Tensor:
        return torch.einsum('...ijkl,...kl->...ij',map,signal)

    @staticmethod
    def warp_spectrum(transform:torch.Tensor, spectrum:torch.Tensor)->torch.Tensor:
        return torch.einsum('...ijkl,...kl->...ij',transform,spectrum)
    
    @staticmethod
    def padding_for_stgw(gw_signal:torch.Tensor,signal:torch.Tensor,window_size:int=16):
        # calculate size
        hop_length = window_size//2
        length = [None,None]
        pad = []
        for i in [-1,-2]:
            length[i] = signal.size(i)
            n_window = (length[i]+hop_length-1)//hop_length
            n_window = n_window//2+1
            whole_length = window_size*n_window+hop_length
            pad = pad+[hop_length,whole_length-length[i]-hop_length]
        length = tuple(length)
        pad = tuple(pad)
        # padding
        signal = torch.nn.functional.pad(signal,pad)
        gw_signal = torch.nn.functional.pad(gw_signal,pad)
        return gw_signal,signal,length
    
    @staticmethod
    def strip_for_stgw(signal:torch.Tensor,length:list,window_size:int=16):
        return signal[...,window_size//2:window_size//2+length[-2],window_size//2:window_size//2+length[-1]]
    
    @staticmethod
    def short_time_warp(gw_signal:torch.Tensor, signal:torch.Tensor, window_size:int=16, n:int=16)->torch.Tensor:
        """short time warp with hanning window

        Args:
            gw_signal (torch.Tensor, axis=(...,x,y))
            signal (torch.Tensor, axis=(...,x,y))
            window_size (int)
            n (int): The iteration number for scaling-squareing

        Returns:
            torch.Tensor: output signal
        """
        # calculate size
        hop_length = window_size//2
        lengthx = signal.size(-2)
        lengthy = signal.size(-1)
        n_windowx = lengthx//window_size
        n_windowy = lengthy//window_size
        # stiride GW signal
        s = gw_signal.stride(-3)
        sx = gw_signal.stride(-2)
        sy = gw_signal.stride(-1)
        # (...,2,n_window,window_size)
        gw_signal = torch.as_strided(
            gw_signal,
            size=gw_signal.size()[:-3]+(2,2,n_windowx,n_windowy,2,window_size,window_size),
            stride=gw_signal.stride()[:-3]+(hop_length*sx,hop_length*sy,window_size*sx,window_size*sy,s,sx,sy)
        )
        # windowing
        window = torch.hann_window(window_size,device=gw_signal.device)
        gw_signal = gw_signal*window[:,None]*window[None,:]/n
        # calculate GW map
        # (...,2,n_window,window_size,window_size)
        gw_map =    gw2.transform_to_map(
                        gw2.operator_to_transform(
                            gw2.spectrum_to_operator(
                                gw2.signal_to_spectrum(gw_signal))))
        # warp
        size1 = signal.size()[:-2]+(n_windowx*window_size,n_windowy*window_size)
        size2 = signal.size()[:-2]+(n_windowx,window_size,n_windowy,window_size)
        for _ in range(n):
            signal = mytorch.cat(
                [
                    [
                        gw2.warp_signal(
                            gw_map[...,0,0,:,:,:,:,:,:],
                            signal[...,:-hop_length,:-hop_length].reshape(size2).transpose(-3,-2)
                        ).transpose(-3,-2).reshape(size1),
                        signal[...,:-hop_length,-hop_length:]
                    ],
                    [
                        signal[...,-hop_length:,:-hop_length],
                        signal[...,-hop_length:,-hop_length:]
                    ],
                ],
                dim=[-2,-1]
            )
            signal = mytorch.cat(
                [
                    [
                        signal[...,:-hop_length,:hop_length],
                        gw2.warp_signal(
                            gw_map[...,0,1,:,:,:,:,:,:],
                            signal[...,:-hop_length,hop_length:].reshape(size2).transpose(-3,-2)
                        ).transpose(-3,-2).reshape(size1)
                    ],
                    [
                        signal[...,-hop_length:,:hop_length],
                        signal[...,-hop_length:,hop_length:]
                    ],
                ],
                dim=[-2,-1]
            )
            signal = mytorch.cat(
                [
                    [
                        signal[...,:hop_length,:hop_length],
                        signal[...,:hop_length,hop_length:],
                    ],
                    [
                        signal[...,hop_length:,:hop_length],
                        gw2.warp_signal(
                            gw_map[...,1,1,:,:,:,:,:,:],
                            signal[...,hop_length:,hop_length:].reshape(size2).transpose(-3,-2)
                        ).transpose(-3,-2).reshape(size1)
                    ],
                ],
                dim=[-2,-1]
            )
            signal = mytorch.cat(
                [
                    [
                        signal[...,:hop_length,:-hop_length],
                        signal[...,:hop_length,-hop_length:],
                    ],
                    [
                        gw2.warp_signal(
                            gw_map[...,1,0,:,:,:,:,:,:],
                            signal[...,hop_length:,:-hop_length].reshape(size2).transpose(-3,-2)
                        ).transpose(-3,-2).reshape(size1),
                        signal[...,hop_length:,-hop_length:]
                    ],
                ],
                dim=[-2,-1]
            )
            
        return signal

    def inverse_short_time_warp(gw_signal:torch.Tensor, signal:torch.Tensor, window_size:int=16, n:int=16)->torch.Tensor:
        """inverse short time warp with hanning window

        Args:
            gw_signal (torch.Tensor, axis=(...,x,y))
            signal (torch.Tensor, axis=(...,x,y))
            window_size (int)
            n (int): The iteration number for scaling-squareing

        Returns:
            torch.Tensor: output signal
        """
        # calculate size
        hop_length = window_size//2
        lengthx = signal.size(-2)
        lengthy = signal.size(-1)
        n_windowx = lengthx//window_size
        n_windowy = lengthy//window_size
        # stiride GW signal
        s = gw_signal.stride(-3)
        sx = gw_signal.stride(-2)
        sy = gw_signal.stride(-1)
        # (...,2,n_window,window_size)
        gw_signal = torch.as_strided(
            gw_signal,
            size=gw_signal.size()[:-3]+(2,2,n_windowx,n_windowy,2,window_size,window_size),
            stride=gw_signal.stride()[:-3]+(hop_length*sx,hop_length*sy,window_size*sx,window_size*sy,s,sx,sy)
        )
        # windowing
        window = torch.hann_window(window_size,device=gw_signal.device)
        gw_signal = -gw_signal*window[:,None]*window[None,:]/n
        # calculate GW map
        # (...,2,n_window,window_size,window_size)
        gw_map =    gw2.transform_to_map(
                        gw2.operator_to_transform(
                            gw2.spectrum_to_operator(
                                gw2.signal_to_spectrum(gw_signal))))
        # warp
        size1 = signal.size()[:-2]+(n_windowx*window_size,n_windowy*window_size)
        size2 = signal.size()[:-2]+(n_windowx,window_size,n_windowy,window_size)
        for _ in range(n):
            signal = mytorch.cat(
                [
                    [
                        signal[...,:hop_length,:-hop_length],
                        signal[...,:hop_length,-hop_length:],
                    ],
                    [
                        gw2.warp_signal(
                            gw_map[...,1,0,:,:,:,:,:,:],
                            signal[...,hop_length:,:-hop_length].reshape(size2).transpose(-3,-2)
                        ).transpose(-3,-2).reshape(size1),
                        signal[...,hop_length:,-hop_length:]
                    ],
                ],
                dim=[-2,-1]
            )
            signal = mytorch.cat(
                [
                    [
                        signal[...,:hop_length,:hop_length],
                        signal[...,:hop_length,hop_length:],
                    ],
                    [
                        signal[...,hop_length:,:hop_length],
                        gw2.warp_signal(
                            gw_map[...,1,1,:,:,:,:,:,:],
                            signal[...,hop_length:,hop_length:].reshape(size2).transpose(-3,-2)
                        ).transpose(-3,-2).reshape(size1)
                    ],
                ],
                dim=[-2,-1]
            )
            signal = mytorch.cat(
                [
                    [
                        signal[...,:-hop_length,:hop_length],
                        gw2.warp_signal(
                            gw_map[...,0,1,:,:,:,:,:,:],
                            signal[...,:-hop_length,hop_length:].reshape(size2).transpose(-3,-2)
                        ).transpose(-3,-2).reshape(size1)
                    ],
                    [
                        signal[...,-hop_length:,:hop_length],
                        signal[...,-hop_length:,hop_length:]
                    ],
                ],
                dim=[-2,-1]
            )
            signal = mytorch.cat(
                [
                    [
                        gw2.warp_signal(
                            gw_map[...,0,0,:,:,:,:,:,:],
                            signal[...,:-hop_length,:-hop_length].reshape(size2).transpose(-3,-2)
                        ).transpose(-3,-2).reshape(size1),
                        signal[...,:-hop_length,-hop_length:]
                    ],
                    [
                        signal[...,-hop_length:,:-hop_length],
                        signal[...,-hop_length:,-hop_length:]
                    ],
                ],
                dim=[-2,-1]
            )
            
            
        return signal
    

if __name__=='__main__':
    import argparse
    import os
    
    from matplotlib import pyplot as plt
    
    parser = argparse.ArgumentParser()
    parser.add_argument('spectrums_path',type=str)
    parser.add_argument('image_dir',type=str)
    args = parser.parse_args()
    
    spectrums = torch.load(args.spectrums_path)
    if spectrums[0].ndim!=1:
        for i,spectrum in enumerate(spectrums):
            fig,ax = plt.subplots(2,2,figsize=(16,16))
            sizex = spectrum.size(-2)
            sizey = spectrum.size(-1)
            signal = torch.zeros(sizex,sizey)
            row = torch.arange(sizex)[:,None]
            col = torch.arange(sizey)[None,:]
            signal[(row//(sizex//32)+col//(sizey//32))%2==0] = 1
            # ax[0,0].imshow(signal)
            window_size = 8
            a = 8/window_size
            gw_signal = gw2.spectrum_to_signal(spectrum)*a**2*2
            ax[1,0].imshow(gw_signal[0].real)
            ax[0,1].imshow(gw_signal[1].real)
            gw_signal,signal,length = gw2.padding_for_stgw(gw_signal,signal,window_size=window_size)
            signal = gw2.short_time_warp(gw_signal.cuda(),signal.cuda(),window_size=window_size,n=64)
            signal_ = gw2.inverse_short_time_warp(gw_signal.cuda(),signal.cuda(),window_size=window_size,n=64)
            signal = gw2.strip_for_stgw(signal,length,window_size=window_size).cpu()
            signal_ = gw2.strip_for_stgw(signal_,length,window_size=window_size).cpu()
            ax[0,0].imshow(signal_,vmin=0,vmax=1)
            ax[1,1].imshow(signal,vmin=0,vmax=1)
            fig.savefig(os.path.join(args.image_dir,f'parameter{i}.png'))
    elif spectrums[0].dtype is torch.cfloat:
        for i,spectrum in enumerate(spectrums):
            size = spectrum.size(-1)
            if size < 1024:
                signal = gw.spectrum_to_signal(spectrum)
                operator = gw.spectrum_to_operator(spectrum,norm=False)
                transform = gw.operator_to_transform(operator)
                ir = gw.transform_to_ir(transform)
                filter = gw.ir_to_filter(ir)
                map = gw.transform_to_map(transform)
                function = gw.filter_to_function(filter)
                print('spectrum-signal inverse error',torch.dist(spectrum,gw.signal_to_spectrum(signal)))
                print('ir-filter inverse error',torch.dist(ir,gw.filter_to_ir(filter)))
                print('transform-filter inverse error',torch.dist(transform,gw.filter_to_transform(filter)))
                print('transform-map inverse error',torch.dist(transform,gw.map_to_transform(map,transform.size(-1)%2==1)))
                print('filter-function inverse error',torch.dist(filter,gw.function_to_filter(function)))
                
                image = torch.zeros((size-1)*2,(size-1)*2)
                row = torch.arange((size-1)*2)[:,None]
                col = torch.arange((size-1)*2)[None,:]
                image[(row//(size//8)+col//(size//8))%2==0] = 1
                
                fig,ax = plt.subplots(3,3,figsize=(15,15))
                ax[0,0].set_title('GW inpulse response')
                ax[0,0].set_xlabel('frequency bin')
                ax[0,0].plot(ir.abs())
                ax[0,1].set_title('GW signal')
                ax[0,1].set_xlabel('time bin')
                ax[0,1].set_xmargin(0)
                ax[0,1].plot(signal)
                ax[0,2].set_title('GW spectrum')
                ax[0,2].set_xlabel('frequency bin')
                ax[0,2].plot(spectrum.abs())
                ax[1,0].set_title('Sample image warped with GW map')
                ax[1,0].imshow(map@image.T,vmin=0,vmax=1,origin='lower')
                ax[1,1].set_title('GW map')
                ax[1,1].set_xlabel('input time bin')
                ax[1,1].set_ylabel('output time bin')
                ax[1,1].imshow(map,origin='lower')
                ax[1,2].set_title('GW transform')
                ax[1,2].set_xlabel('input frequency bin')
                ax[1,2].set_ylabel('output frequency bin')
                ax[1,2].imshow(transform.abs(),origin='lower')
                ax[2,0].set_title('GW function')
                ax[2,0].set_xlabel('input time')
                ax[2,0].set_ylabel('output time')
                ax[2,0].set_xmargin(0)
                ax[2,0].set_ymargin(0)
                ax[2,0].plot(function,torch.linspace(0,1,function.size(-1)))
                ax[2,1].set_title('Sample image warped with GW function')
                ax[2,1].imshow(image[function.mul((size-1)*2).minimum(torch.tensor(image.size(0)-1)).long()],origin='lower')
                ax[2,2].set_title('GW operator')
                ax[2,2].set_xlabel('input frequency bin')
                ax[2,2].set_ylabel('output frequency bin')
                ax[2,2].imshow(operator.abs(),origin='lower')
                fig.savefig(os.path.join(args.image_dir,f'parameter{i}.png'))
            else:
                window_size = 16
                fig,ax = plt.subplots(1,2,figsize=(16,8))
                gw_signal = gw.spectrum_to_signal(spectrum)[None,None,:]*128*(32/window_size)**2
                ax[0].plot(gw_signal[0,0])
                signal = torch.eye(gw_signal.size(-1))[:,None,:]
                gw_signal,signal,length = gw.padding_for_stgw(gw_signal,signal,window_size=window_size)
                signal = gw.short_time_warp(gw_signal.cuda(),signal.cuda(),window_size=window_size,n=16)
                signal = gw.strip_for_stgw(signal,length,window_size=window_size)[:,0,:].cpu()
                ax[1].imshow(signal.T,origin='lower')
                fig.savefig(os.path.join(args.image_dir,f'parameter{i}.png'))
    else:
        for i,spectrum in enumerate(spectrums):
            size = spectrum.size(-1)
            signal = rgw.spectrum_to_signal(spectrum)
            operator = rgw.spectrum_to_operator(spectrum,norm=False)
            transform = rgw.operator_to_transform(operator)
            ir = rgw.transform_to_ir(transform)
            filter = rgw.ir_to_filter(ir)
            map = rgw.transform_to_map(transform)
            function = rgw.filter_to_function(filter)
            print('spectrum-signal inverse error',torch.dist(spectrum,rgw.signal_to_spectrum(signal)))
            print('ir-filter inverse error',torch.dist(ir,rgw.filter_to_ir(filter)))
            print('transform-filter inverse error',torch.dist(transform,rgw.filter_to_transform(filter)))
            print('transform-map inverse error',torch.dist(transform,rgw.map_to_transform(map)))
            print('filter-function inverse error',torch.dist(filter,rgw.function_to_filter(function)))
            
            image = torch.zeros(size,size)
            row = torch.arange(size)[:,None]
            col = torch.arange(size)[None,:]
            image[(row//(size//32)+col//(size//32))%2==0] = 1
            
            fig,ax = plt.subplots(3,3,figsize=(15,15))
            ax[0,0].set_title('GW inpulse response')
            ax[0,0].set_xlabel('frequency bin')
            ax[0,0].plot(ir)
            ax[0,1].set_title('GW signal')
            ax[0,1].set_xlabel('time bin')
            ax[0,1].set_xmargin(0)
            ax[0,1].plot(signal)
            ax[0,2].set_title('GW spectrum')
            ax[0,2].set_xlabel('frequency bin')
            ax[0,2].plot(spectrum)
            ax[1,0].set_title('Sample image warped with GW map')
            ax[1,0].imshow(map@image.T,vmin=0,vmax=1,origin='lower')
            ax[1,1].set_title('GW map')
            ax[1,1].set_xlabel('input time bin')
            ax[1,1].set_ylabel('output time bin')
            ax[1,1].imshow(map,origin='lower')
            ax[1,2].set_title('GW transform')
            ax[1,2].set_xlabel('input frequency bin')
            ax[1,2].set_ylabel('output frequency bin')
            ax[1,2].imshow(transform,origin='lower')
            ax[2,0].set_title('GW function')
            ax[2,0].set_xlabel('input time')
            ax[2,0].set_ylabel('output time')
            ax[2,0].set_xmargin(0)
            ax[2,0].set_ymargin(0)
            ax[2,0].plot(function,torch.linspace(0,1,function.size(-1)))
            ax[2,1].set_title('Sample image warped with GW function')
            ax[2,1].imshow(image[function.mul(size).minimum(torch.tensor(image.size(0)-1)).long()],origin='lower')
            ax[2,2].set_title('GW operator')
            ax[2,2].set_xlabel('input frequency bin')
            ax[2,2].set_ylabel('output frequency bin')
            ax[2,2].imshow(operator,origin='lower')
            fig.savefig(os.path.join(args.image_dir,f'parameter{i}.png'))
            
        
        


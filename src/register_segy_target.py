#%%
"""
Example script to register two volumes with VoxelMorph models

Please make sure to use trained models appropriately. 
Let's say we have a model trained to register subject (moving) to atlas (fixed)
One could run:

python register.py --gpu 0 /path/to/test_vol.nii.gz /path/to/atlas_norm.nii.gz --out_img /path/to/out.nii.gz --model_file ../models/cvpr2018_vm2_cc.h5 
"""

#%%
# py imports
import os
import sys
from argparse import ArgumentParser

# third party
import tensorflow as tf
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

import matplotlib.pyplot as plt

#%%
# project
sys.path.append('/home/jdram/voxelmorph/src')
import networks, losses
sys.path.append('/home/jdram/voxelmorph/ext/neuron')
import neuron.layers as nrn_layers

#%%

def register(gpu_id, mov, fix, model_file, out_img, out_warp):
    """
    register moving and fixed. 
    """  
    #assert model_file, "A model file is necessary"
    #assert out_img or out_warp, "output image or warp file needs to be specified"

    # GPU handling
    if gpu_id is not None:
        gpu = '/gpu:' + str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        gpu = '/cpu:0'

    # load data
    #mov_nii = nib.load(moving)
    #mov = mov_nii.get_data()[np.newaxis, ..., np.newaxis]
    #fix_nii = nib.load(fixed)
    #fix = fix_nii.get_data()[np.newaxis, ..., np.newaxis]

    with tf.device(gpu):
        # load model
        loss_class = losses.Miccai2018(0.02, 10, flow_vol_shape=[256-64])
        custom_objects = {'SpatialTransformer': nrn_layers.SpatialTransformer,
                 'VecInt': nrn_layers.VecInt,
                 'Sample': networks.Sample,
                 'Rescale': networks.RescaleDouble,
                 'Resize': networks.ResizeDouble,
                 'Negate': networks.Negate,
                 'recon_loss': loss_class.recon_loss, # values shouldn't matter
                 'kl_loss': loss_class.kl_loss        # values shouldn't matter
                 }


        net = keras.models.load_model(model_file, custom_objects=custom_objects)
        
        # register
        [moved, warp] = net.predict([mov, fix])

    return moved, warp

def plt_commons(title, ix, cbar_label=None, pre='x', name='x', w=0, suff=''):
    plt.title(title.title())
    plt.ylabel('Time [s]')
    plt.xlabel(f'{ix} [m]')
    if cbar_label:
        cbar = plt.colorbar(pad=0.1, orientation="horizontal", aspect=20)
        cbar.set_label(cbar_label)
    plt.savefig(f'{pre}_{ix}_{title}_{w}{suff}.png'.replace(' ','_').lower(), bbox_inches='tight')

def warp_results(moving, fixed, moved, warped, pre='x'):
    """
    Warp Results and plot on CPU
    moving: monitor
    fixed: base
    moved: matched monitor
    warped: warp vector
    pre: extra suffix for multiple experiments
    """
    seis = np.max([np.abs(moving[0, 32, :, :, 0]).max(), \
        np.abs(fixed[0, 32, :, :, 0]).max(), \
        np.abs(moved[0, 32, :, :, 0]).max(), \
        np.abs(moving[0, :, 32, :, 0]).max(), \
        np.abs(fixed[0, :, 32, :, 0]).max(), \
        np.abs(moved[0, :, 32, :, 0]).max(), \
    ])

    orig_warp = warped.copy()
    orig_warp[:,:,:,:,3:] = np.exp(orig_warp[:,:,:,:,3:]/2)

    warped[:,:,:,:,3:] = np.exp(warped[:,:,:,:,3:]/2)
    warped[:,:,:,:,0:2] *= 12.5
    warped[:,:,:,:,3:5] *= 12.5
    warped[:,:,:,:,2] *= 0.004 * 1e3
    warped[:,:,:,:,5] *= 0.004 * 1e3
    warp_u = np.max([np.abs(warped[0, 32, :, :, :2]), \
            np.abs(warped[0, :, 32, :, :2])])
    
    warp_ut = np.max([np.abs(warped[0, 32, :, :, 2]), \
            np.abs(warped[0, :, 32, :, 2])])

    diff = np.max([np.abs(moving[0, 32, :, :, 0]-fixed[0, 32, :, :, 0]).max(), \
        np.abs(moved[0, 32, :, :, 0]-fixed[0, 32, :, :, 0]).max(), \
    ])
    
    for name, dat in {'monitor': moving, 'base': fixed, 'matched': moved, 'warp': warped}.items():
        extent = (0,64*12.5, (256-64)*.004, 0)
        plt_size= (3,9)

        plt_args = {'extent':extent, 'aspect':'auto', 'interpolation':'bicubic'}
        
        for w in range(dat.shape[-1]):
            
            if w < 3:
                cmap = 'RdBu'
        
                if dat.shape[-1] == 1:
                    va = diff
                    cb = 'Amplitude'

                    # Difference Images
                    if not 'base' in name:
                        plt.figure(figsize=plt_size)
                        plt.imshow(dat[0, 32, :, :, w].T-fixed[0, 32, :, :, w].T, cmap=cmap, vmin=-va, vmax=va, **plt_args) 
                        plt_commons(name+' difference', 'inline', cb, pre=pre, name=name, w=w, suff='_diff')

                        plt.figure(figsize=plt_size)
                        plt.imshow(dat[0, :, 32, :, w].T-fixed[0, :, 32, :, w].T, cmap=cmap, vmin=-va, vmax=va, **plt_args) 
                        plt_commons(name+' difference', 'crossline', cb, pre=pre, name=name, w=w, suff='_diff')
                    va = seis
                
                else:
                    va = warp_u
                    cb = 'Spatial Shift [m]'
                    if w == 2:
                        va = warp_ut
                        cb = 'Time Shift [ms]'
                
                # Intentionally left unindented
                plt.figure(figsize=plt_size)
                plt.imshow(dat[0, 32, :, :, w].T, cmap=cmap, vmin=-va, vmax=va, **plt_args) 
                plt_commons(name, 'inline', cb, pre=pre, name=name, w=w)
        
                plt.figure(figsize=plt_size)
                plt.imshow(dat[0, :, 32, :, w].T, cmap=cmap, vmin=-va, vmax=va, **plt_args) 
                plt_commons(name, 'crossline', cb, pre=pre, name=name, w=w)

            else:
                cmap = 'viridis'
                cb = 'Spatial Uncertainty $1\sigma$ [m]'
                va = warp_u
                if w == 5:
                    cb = 'Temporal Uncertainty $1\sigma$ [ms]'
                    va = warp_ut
                
                plt.figure(figsize=plt_size)
                plt.imshow(dat[0, 32, :, :, w].T,cmap=cmap, **plt_args)
                plt_commons(name, 'inline', cb, pre=pre, name=name, w=w)

                plt.figure(figsize=plt_size)
                plt.imshow(dat[0, :, 32, :, w].T,cmap=cmap, **plt_args)
                plt_commons(name, 'crossline', cb, pre=pre, name=name, w=w)

            if w == 0:
                np.save(f'{name}_{pre}.npy', dat)
        plt.close('all')

#%%
vm_dir = '/home/jdram/voxelmorph/'
model_path = "/home/jdram/voxelmorph/models/backup/miccai_300_full_deep.h5" #64x64x192 full unet
#model_path = "/home/jdram/voxelmorph/models/backup/miccai_e290_64-64-192_upsample.h5" #64x64x192 upsampled

monitor_hfd = np.load(os.path.join("/", "scratch", "jdram", "voxelmorph", "halfdan", "ts12_hfd_h05_fin_o_trim_adpc_002582_cube.npy"))
base_hfd    = np.load(os.path.join("/", "scratch", "jdram", "voxelmorph", "halfdan", "ts12_hfd_h93_fin_o_trim_adpc_002568_cube.npy"))

#k = 350
#p = 850
#q = 475
k = 350
p = 200
q = 475
moving = monitor_hfd[np.newaxis, k-32:k+32,p:p+64,q:q+256-64,np.newaxis]
fixed  =    base_hfd[np.newaxis,k-32:k+32,p:p+64,q:q+256-64,np.newaxis]

#%%
moved, warped = register(None, moving, fixed, model_path, None, None)

#%%
warp_results(moving,fixed,moved,warped,'hfd')

#%%

monitor = np.load(os.path.join(vm_dir, "data","ts12_dan_a05_fin_o_trim_adpc_002682_cube.npy"))
base    = np.load(os.path.join(vm_dir, "data","ts12_dan_a88_fin_o_trim_adpc_002661_cube.npy"))

k=225
q=350
p=300

moving = monitor[np.newaxis, k-32:k+32,p:p+64,q:q+256-64,np.newaxis]
fixed  =    base[np.newaxis,k-32:k+32,p:p+64,q:q+256-64,np.newaxis]

#%%
moved, warped = register(None, moving, fixed, model_path, None, None)

#%%

#np.save("moving_a.npy",moving)
#np.save("fixed_a.npy",fixed)
#np.save("moved_a.npy",moved)
#np.save("warped_a.npy",warped)

warp_results(moving,fixed,moved,warped,'a')



base2    = np.load(os.path.join(vm_dir, "data","ts12_dan_d05_fin_o_trim_adpc_002696_cube.npy"))
monitor2 = np.load(os.path.join(vm_dir, "data","ts12_dan_d12_fin_o_trim_adpc_002710_cube.npy"))

moving = monitor2[np.newaxis, k-32:k+32,p:p+64,q:q+256-64,np.newaxis]
fixed  =    base2[np.newaxis,k-32:k+32,p:p+64,q:q+256-64,np.newaxis]

#%%
moved, warped = register(None, moving, fixed, model_path, None, None)

#%%
warp_results(moving,fixed,moved,warped,'d')


moving = monitor2[np.newaxis, k-32:k+32,p:p+64,q:q+256-64,np.newaxis]
fixed  =    monitor[np.newaxis,k-32:k+32,p:p+64,q:q+256-64,np.newaxis]

#%%
moved, warped = register(None, moving, fixed, model_path, None, None)

#%%
warp_results(moving,fixed,moved,warped,'ad')

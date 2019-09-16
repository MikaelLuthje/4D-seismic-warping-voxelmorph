import click
import glob
from tqdm import tqdm
import segyio
import numpy as np
from numpy.lib import stride_tricks
from time import time

def cutup(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6#.reshape(-1, *blck)

for filename in glob.iglob("../data/*.sgy"):
    start_time = time()
    numpy_path = filename[:-3]+"npy"
    cube_path = filename[:-4]+"_cube.npy"
    print(f"=== Converting {filename[:-4].split('/')[-1]} ===")
    print("Opening file")
    with segyio.open(filename, 'r', strict=False) as segy:
        print("Precomputing Statistics")
        f_time = time()
        ilines=np.unique(segy.attributes(segyio.TraceField.INLINE_3D)[:])
        mi = min(ilines)
        mia = max(ilines)
        il = len(np.unique(ilines))
        xlines=np.unique(segy.attributes(segyio.TraceField.CROSSLINE_3D)[:])
        mx = min(xlines)
        mxa = max(xlines)
        xl = len(np.unique(xlines))
        out = np.full((il-600, xl-600, len(segy.samples)),np.nan,dtype=np.float32)
        print(f" - in {time() - f_time:.2f} seconds")
        print("Extracting Traces")
        f_time = time()
        for x in tqdm(range(len(segy.trace))):
            i = segy.attributes(segyio.TraceField.INLINE_3D)[x]-mi
            c = segy.attributes(segyio.TraceField.CROSSLINE_3D)[x]-mx
            if (i >= 300) and (i < il - 300) and (c >= 300) and (c < xl - 300):
                out[i-300, c-300, :] = segy.trace[x]
    print(f" - in {time() - f_time:.2f} seconds")
    
    print("Normalization")
    f_time = time()
    out_clip = np.percentile(np.abs(out), 99)
    out = np.clip(out, -out_clip, out_clip)
    out /= max(abs(out.min()), out.max())
    print(f" - in {time() - f_time:.2f} seconds")
    
    print("Reshaping Data to Batches")
    f_time = time()
    i, j, k = (64, 64, 64) # Batch Dimension
    s_i, s_j, s_k = (2, 2, 1.2) # Strides
    y = cutup(out, (i, j, k), (int(i/s_i), int(j/s_j), int(k/s_k))).reshape(-1 , i, j, k, 1)
    print(f" - in {time() - f_time:.2f} seconds")
    
    print("Saving File")
    f_time = time()
    np.save(numpy_path, y)
    np.save(cube_path, out)
    print(f" - in {time() - f_time:.2f} seconds")

    print("Testing Load")
    f_time = time()
    z = np.load(numpy_path)
    np.testing.assert_array_equal(y,z)
    del y
    del z
    z = np.load(cube_path)
    np.testing.assert_array_equal(out,z)
    del z 
    print(f" - in {time() - f_time:.2f} seconds")
    print("--- --- ---")
    print(f"Total: {time() - start_time:.2f} seconds")
    print("=== === ===", end="\n\n")
    
print("Conversion Completed.")

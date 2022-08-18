import df_to_grid
import pandas as pd
import pathlib
import tifffile, zarr
import numpy as np

path_table = r"Z:\RareCyte-S3\YC-analysis\P37_CRCstudy_Round1\ml-jerry_cell_types\pixel_coors\Celltable_C2.csv"
path_mask = r"Z:\RareCyte-S3\YC-analysis\P37_CRCstudy_Round1\P37_S30-CRC02\segmentation\P37_S30_A24_C59kX_E15@20220106_014319_409148\nucleiRingMask.tif"
path_img = r"X:\crc-scans\histowiz scans\registered-orion\18459$LSP10364$US$SCAN$OR$001 _092347-registered.ome.tif"
dir_output = r"Z:\RareCyte-S3\YC-analysis\P37_CRCstudy_Round1\P37_S30-CRC02\ml\test"

PREFIX = 'CRC02'
SIZE = 256
COLUMN = 'Cat1'
SIMILARITY = [0.5, 1, 1]
MIN_COUNTS = 50
MIN_SIMILARITY = 0.92
MAX_SIMILARITY = 1
Y_COLUMN_NAME = 'Y_centroid'
X_COLUMN_NAME = 'X_centroid'


# create output directories
import pathlib
out_dir = pathlib.Path(dir_output)
(out_dir / 'mask').mkdir(exist_ok=True, parents=True)
(out_dir / 'img').mkdir(exist_ok=True, parents=True)


# read single-cell table and select patches based on cosine similarity criteria
df = pd.read_csv(path_table).set_index('CellID')
grid_df = df_to_grid.GridDf(
    df,
    grid_size=SIZE,
    y_column_name=Y_COLUMN_NAME,
    x_column_name=X_COLUMN_NAME
)
grid_df.grid_category_count(COLUMN)
df_similarity = grid_df.calculate_similarity(COLUMN, SIMILARITY)
df_selected = (
    df_similarity
        .query('Counts >= @MIN_COUNTS')
        .query('Similarity >= @MIN_SIMILARITY')
        .query('Similarity <= @MAX_SIMILARITY')
)


# read mask patches
mask = zarr.open(tifffile.TiffFile(path_mask).series[0].aszarr())
masks = df_selected.apply(
    lambda x: mask[int(x['row_s']):int(x['row_s'])+SIZE, int(x['col_s']):int(x['col_s'])+SIZE],
    axis=1
)


# recolor mask with class labels
def mapping_indexer(
    df, value_for_missing_key=0, cat_column_name='Cat1'
):
    indexer = np.ones(df.index.max() + 1) * value_for_missing_key
    indexer[df.index.values] = df[cat_column_name].values
    return indexer

mask_indexer = mapping_indexer(df, cat_column_name=COLUMN)
recolored = masks.apply(lambda x: mask_indexer[x].astype(np.uint8))


# write recolored masks to mask/
for i in range(len(df_selected)):
    img = recolored.iloc[i]
    row_s, col_s = df_selected.iloc[i][['row_s', 'col_s']].astype(int)
    tifffile.imsave(out_dir / 'mask' / f"{PREFIX}-{COLUMN}-rs_{row_s}-cs_{col_s}.tif", img)


# write patch coordinate table to `dir_output`
df_selected.to_csv(out_dir / f"{PREFIX}-{COLUMN}-selected_patches.csv")


# read and write H&E patches to img/
from joblib import Parallel, delayed
import skimage.util

def wrap_save(tiffpath, rs, cs, size, out_dir):
    t = tifffile.TiffFile(tiffpath)
    z = zarr.open(t.series[0].levels[0].aszarr())[0]
    i = skimage.util.img_as_float64(z[:, rs:rs+size, cs:cs+size])
    tifffile.imsave(
        out_dir / 'img' / f"{PREFIX}-rs_{rs}-cs_{cs}.tif",
        skimage.util.img_as_ubyte(i)
        # i
    )
    t.close()
    return 

_ = Parallel(n_jobs=1, verbose=1)(delayed(wrap_save)(
    path_img,
    i[0],
    i[1],
    SIZE,
    out_dir
) for i in df_selected[['row_s', 'col_s']].values)


# remove blurry image
import scipy.ndimage
import skimage.restoration
# Pre-calculate the Laplacian operator kernel. We'll always be using 2D images.
_laplace_kernel = skimage.restoration.uft.laplacian(2, (3, 3))[1]

def whiten(img, sigma):
    img = skimage.img_as_float32(img)
    if sigma == 0:
        output = scipy.ndimage.convolve(img, _laplace_kernel)
    else:
        output = scipy.ndimage.gaussian_laplace(img, sigma)
    return output

img_files = sorted(pathlib.Path(out_dir / 'img').glob('*.tif'))
mask_files = sorted(pathlib.Path(out_dir / 'mask').glob('*.tif'))

assert len(img_files) == len(mask_files)

for ip, mp in zip(img_files, mask_files):
    img = tifffile.imread(ip)[1]
    if whiten(np.where(img == 0, img.max(), img), 0).var() < 1e-3:
        ip.unlink()
        mp.unlink()
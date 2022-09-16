import df_to_grid
import pandas as pd
import pathlib
import tifffile, zarr
import numpy as np
import pathlib
import tqdm


path_table = r"W:\crc-scans\C1-C40-sc-tables\P37_S29-CRC01\quantification\P37_S29_A24_C59kX_E15@20220106_014304_946511_cellRingMask_morphology.csv"
path_mask = r"W:\crc-scans\C1-C40-segmentation-masks\P37_S29-CRC01\segmentation\P37_S29_A24_C59kX_E15@20220106_014304_946511\nucleiRingMask.tif"
path_img_he = r"X:\crc-scans\histowiz scans\registered-orion\18459$LSP10353$US$SCAN$OR$001 _093059-registered.ome.tif"
path_img_if = r"Z:\RareCyte-S3\P37_CRCstudy_Round1\P37_S29_A24_C59kX_E15@20220106_014304_946511.ome.tiff"
path_qc_bad_ids = r"W:\crc-scans\C1-C40-cylinter\output2\noisy_roi_ids\noisy_roi_ids.csv"
dir_output = r'W:\crc-scans\C1-C40-patches\CRC01'


PREFIX = 'CRC01'
SIZE = 224
COLUMN = 'Pass QC'
MIN_COUNTS = 30
Y_COLUMN_NAME = 'Y_centroid'
X_COLUMN_NAME = 'X_centroid'
N_PATCHES = 3000


# create output directories
out_dir = pathlib.Path(dir_output)
(out_dir / 'mask').mkdir(exist_ok=True, parents=True)
(out_dir / 'img_if').mkdir(exist_ok=True, parents=True)
(out_dir / 'img_he').mkdir(exist_ok=True, parents=True)


# read single-cell table and select patches based on cosine similarity criteria
df = pd.read_csv(path_table).set_index('CellID')
df_qc = pd.read_csv(path_qc_bad_ids).query('Sample == 1')

# add pass qc annotation to each row
df['Pass QC'] = True
df.loc[df_qc['CellID'], 'Pass QC'] = False

# bin the dataframe into grid using SIZE as bin width and get counts
grid_df = df_to_grid.GridDf(
    df,
    grid_size=SIZE,
    y_column_name=Y_COLUMN_NAME,
    x_column_name=X_COLUMN_NAME
)
grid_df_counts = grid_df.grid_category_count(COLUMN)

# filter grid dataframe based on qc annotation
df_selected = (
    grid_df_counts
        .query('Counts >= @MIN_COUNTS')
        .query('`Pass QC_False` == 0')
)


# write mask patches
mask = zarr.open(tifffile.imread(path_mask, aszarr=True, level=0))
h, w = mask.shape
df_selected_sampled = (
    df_selected
        .query('row_s <= @h - @SIZE')
        .query('col_s <= @w - @SIZE')
).sample(N_PATCHES, random_state=10001)
for r, c in tqdm.tqdm(df_selected_sampled[['row_s', 'col_s']].values, ascii=True):
    m  = mask[r:r+SIZE, c:c+SIZE]
    tifffile.imsave(out_dir / 'mask' / f"{PREFIX}-rs_{r}-cs_{c}.tif", m)

# write IF patches
orion = zarr.open(tifffile.imread(path_img_if, aszarr=True, level=0))
for r, c in tqdm.tqdm(df_selected_sampled[['row_s', 'col_s']].values, ascii=True):
    m  = orion[:, r:r+SIZE, c:c+SIZE]
    tifffile.imsave(out_dir / 'img_if' / f"{PREFIX}-rs_{r}-cs_{c}.tif", m)

# write H&E patches
he = zarr.open(tifffile.imread(path_img_he, aszarr=True, level=0))
for r, c in tqdm.tqdm(df_selected_sampled[['row_s', 'col_s']].values, ascii=True):
    m  = he[:, r:r+SIZE, c:c+SIZE]
    tifffile.imsave(out_dir / 'img_he' / f"{PREFIX}-rs_{r}-cs_{c}.tif", m)


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

img_he_files = sorted(pathlib.Path(out_dir / 'img_he').glob('*.tif'))
img_if_files = sorted(pathlib.Path(out_dir / 'img_if').glob('*.tif'))
mask_files = sorted(pathlib.Path(out_dir / 'mask').glob('*.tif'))

assert len(img_he_files) == len(img_if_files)
assert len(img_he_files) == len(mask_files)

for ih, ii, mp in tqdm.tqdm(zip(img_he_files, img_if_files, mask_files), ascii=True):
    img = tifffile.imread(ih)[1]
    if whiten(np.where(img == 0, img.max(), img), 0).var() < 1e-3:
        ih.unlink()
        ii.unlink()
        mp.unlink()
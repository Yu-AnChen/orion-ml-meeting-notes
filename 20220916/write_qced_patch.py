import df_to_grid
import pandas as pd
import pathlib
import tifffile, zarr
import numpy as np
import pathlib
import tqdm


from file_vars import (
    path_table,
    path_mask,
    path_img_he,
    path_img_if,
    path_qc_bad_ids,
    dir_output,
    PREFIX,
    SAMPLE_ID
)


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
df_qc = pd.read_csv(path_qc_bad_ids).query(f"Sample == {SAMPLE_ID}")

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
    tifffile.imwrite(out_dir / 'mask' / f"{PREFIX}-rs_{r}-cs_{c}.tif", m)

# write IF patches
orion = zarr.open(tifffile.imread(path_img_if, aszarr=True, level=0))
for r, c in tqdm.tqdm(df_selected_sampled[['row_s', 'col_s']].values, ascii=True):
    m  = orion[:, r:r+SIZE, c:c+SIZE]
    tifffile.imwrite(out_dir / 'img_if' / f"{PREFIX}-rs_{r}-cs_{c}.tif", m)

# write H&E patches
he = zarr.open(tifffile.imread(path_img_he, aszarr=True, level=0))
for r, c in tqdm.tqdm(df_selected_sampled[['row_s', 'col_s']].values, ascii=True):
    m  = he[:, r:r+SIZE, c:c+SIZE]
    tifffile.imwrite(out_dir / 'img_he' / f"{PREFIX}-rs_{r}-cs_{c}.tif", m)



# remove blurry image and shifted image pairs
import scipy.ndimage
import skimage.restoration
import skimage.registration
# Pre-calculate the Laplacian operator kernel. We'll always be using 2D images.
_laplace_kernel = skimage.restoration.uft.laplacian(2, (3, 3))[1]

def whiten(img, sigma):
    img = skimage.img_as_float32(img)
    if sigma == 0:
        output = scipy.ndimage.convolve(img, _laplace_kernel)
    else:
        output = scipy.ndimage.gaussian_laplace(img, sigma)
    return output

def register(img1, img2, sigma, upsample=1):
    img1w = whiten(img1, sigma)
    img2w = whiten(img2, sigma)
    return skimage.registration.phase_cross_correlation(
        img1w,
        img2w,
        upsample_factor=upsample,
        normalization=None,
        return_error=False,
    )

img_he_files = sorted(pathlib.Path(out_dir / 'img_he').glob('*.tif'))
img_if_files = sorted(pathlib.Path(out_dir / 'img_if').glob('*.tif'))
mask_files = sorted(pathlib.Path(out_dir / 'mask').glob('*.tif'))

assert len(img_he_files) == len(img_if_files)
assert len(img_he_files) == len(mask_files)

for ih, ii, mp in tqdm.tqdm(zip(img_he_files, img_if_files, mask_files), ascii=True):
    img = tifffile.imread(ih)[1]
    img2 = tifffile.imread(ii, key=0)
    if whiten(np.where(img == 0, img.max(), img), 0).var() < 1e-3:
        ih.unlink()
        ii.unlink()
        mp.unlink()
        continue
    if np.linalg.norm(register(img, img2, 1)) > (1 / 0.325):
        ih.unlink()
        ii.unlink()
        mp.unlink()

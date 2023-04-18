import pandas as pd
import numpy as np

import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def binned_coords(
    df, bin_size,
    spatial_x_name='Xt', spatial_y_name='Yt',
):
    # snap coordinates to grid
    df_coords = df[[spatial_y_name, spatial_x_name]] / bin_size
    df_coords = df_coords.apply(lambda x: pd.to_numeric(x.round(), downcast='integer'))
    return df_coords


def pca_channels(df, columns, n_pcs=10, transform_func=None, standardize=True):
    if transform_func is None:
        transform_func = lambda x: x
    if columns is None:
        columns = df.columns

    df_channels = df[columns].transform(transform_func)
    scaled_data = df_channels
    if standardize:
        scaled_data = sklearn.preprocessing.scale(df_channels)
    pca = PCA(n_components=n_pcs, random_state=1001)
    pca.fit(scaled_data)
    print(np.cumsum(pca.explained_variance_ratio_))
    return pca


def gmm_cluster_by_pcs(
    df, columns,
    bin_size=224,
    spatial_x_name='X_centroid', spatial_y_name='Y_centroid',
    transform_func=None,
    n_pcs=10, n_components=20,
    standardize=True,
    viz=False
):
    if transform_func is None:
        transform_func = lambda x: x
    df_coords = binned_coords(df, bin_size, spatial_x_name, spatial_y_name)
    binned_df = (
        df[columns]
        .transform(transform_func)
        .groupby([df_coords[spatial_y_name], df_coords[spatial_x_name]])
        .mean()
    )
    # rescaled features seems to give cleaner result
    pca = pca_channels(
        binned_df, columns,
        n_pcs=n_pcs,
        transform_func=None,
        standardize=standardize
    )
    pcs = pca.transform(binned_df)

    clusters = GaussianMixture(
        n_components=n_components, random_state=1001
    ).fit_predict(pcs)
    
    # kmeans seems to give noisy result
    # from sklearn.cluster import KMeans
    # clusters = KMeans(
    #     n_clusters=n_components, random_state=1001
    # ).fit_predict(pcs)

    binned_df['cluster'] = clusters
    if viz:
        plot_cluster(binned_df, column_name='cluster', bg_value=-1, cmap='tab20')
    return binned_df


def gmm_cluster_by_intensity(
    df, columns,
    bin_size=224,
    spatial_x_name='X_centroid', spatial_y_name='Y_centroid',
    transform_func=None,
    n_components=20,
    standardize=True,
    viz=False
):
    if transform_func is None:
        transform_func = np.array
    if columns is None:
        columns = df.columns
    
    df_coords = binned_coords(df, bin_size, spatial_x_name, spatial_y_name)
    binned_df = (
        df[columns]
        .transform(transform_func)
        .groupby([df_coords[spatial_y_name], df_coords[spatial_x_name]])
        .mean()
    )
    scaled_data = binned_df
    if standardize:
        scaled_data = sklearn.preprocessing.scale(binned_df)
    clusters = GaussianMixture(
        n_components=n_components, random_state=1001
    ).fit_predict(scaled_data)
    # kmeans seems to give noisy result
    # from sklearn.cluster import KMeans
    # clusters = KMeans(
    #     n_clusters=n_components, random_state=1001
    # ).fit_predict(pc_raw)
    binned_df['cluster'] = clusters
    if viz:
        plot_cluster(binned_df, column_name='cluster', bg_value=-1, cmap='tab20')
    return binned_df


def df2img(df, column_name, bg_value=0):
    coords = df.index.to_frame().values
    h, w = coords.max(axis=0) + 1
    y, x = coords.T

    img = bg_value * np.ones((h, w))
    img[y, x] = df[column_name]
    return img


def plot_cluster(df, column_name='cluster', bg_value=-1, cmap='tab20', ax=None):
    import matplotlib.pyplot as plt
    img = df2img(df, column_name, bg_value)
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(
        np.where(img == -1, np.nan, img),
        cmap='tab20',
        interpolation='nearest'
    )
    return ax.get_figure()


valid_markers = [
    'Hoechst', 'AF1', 'CD31', 'CD45', 'CD68', 'CD4', 'FOXP3',
    'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163', 'E-cadherin',
    'PD-1', 'Ki67', 'Pan-CK', 'SMA',
]

SIZE = 224
N_PATCHES_PER_CLASS = 1100

import pathlib
import pandas as pd

files = pd.read_csv('files.csv')

for _, row in files.iloc[:10].iterrows():
    PREFIX = row['Name']
    # create output directories
    out_dir = pathlib.Path(
        r'W:\crc-scans\C1-C40-patches\20k'
    ) / row['Name']
    (out_dir / 'mask').mkdir(exist_ok=True, parents=True)
    (out_dir / 'img_if').mkdir(exist_ok=True, parents=True)
    (out_dir / 'img_he').mkdir(exist_ok=True, parents=True)

    path_table = row['Quantification table']
    path_img_he = row['H&E GT450 filepath']
    path_img_if = row['Orion filepath']

    if next((out_dir / 'img_he').iterdir(), None) is not None:
        print(out_dir, 'already processed')
        continue

    print('processing', out_dir)
    
    df = pd.read_csv(path_table)
    grid_df = gmm_cluster_by_pcs(df, valid_markers, transform_func=np.log1p, viz=True)
    # drop tiles touching right and bottom edge
    # so one doesn't have to handle out-of-bound croppings
    row_max, col_max = grid_df.index.max()
    grid_df = grid_df.loc[(slice(0, row_max-1), slice(0, col_max-1)), :]

    sampled_grid_df = grid_df.groupby('cluster').apply(
        lambda x: x.sample(N_PATCHES_PER_CLASS, random_state=1001) 
        if x.index.size >= N_PATCHES_PER_CLASS 
        else x.sample(x.index.size, random_state=1001)
    )

    crop_coords = sampled_grid_df.sort_index(level=[1, 2]).index.to_frame()
    crop_coords.loc[:, ['Y_centroid', 'X_centroid']] *= SIZE

    # write crop_coords to disk
    crop_coords.to_csv(out_dir / 'selected_tiles.csv', index=False)


    import tifffile
    import zarr
    from joblib import Parallel, delayed

    N_JOBS = 12
    orion = zarr.open(tifffile.imread(path_img_if, aszarr=True, level=0))

    n_patches = crop_coords.shape[0]


    def write_patch(df, in_path, out_path, size=SIZE, prefix=PREFIX):
        zimg = zarr.open(tifffile.imread(in_path, aszarr=True, level=0))
        for r, c in zip(df['Y_centroid'], df['X_centroid']):
            img = zimg[:, r:r+size, c:c+size]
            tifffile.imwrite(out_path / f"{prefix}-rs_{r}-cs_{c}.tif", img)
        return 

    _ = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(write_patch)(
        crop_coords.iloc[i:i+n_patches // N_JOBS + 1],
        path_img_he,
        out_dir / 'img_he',
    ) for i in range(0, n_patches, n_patches // N_JOBS + 1))

    _ = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(write_patch)(
        crop_coords.iloc[i:i+n_patches // N_JOBS + 1],
        path_img_if,
        out_dir / 'img_if',
    ) for i in range(0, n_patches, n_patches // N_JOBS + 1))


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

    assert len(img_he_files) == len(img_if_files)

    def remove_bad_tiles(p1, p2):
        img = tifffile.imread(p1)[1]
        img2 = tifffile.imread(p2, key=0)
        if whiten(np.where(img == 0, img.max(), img), 0).var() < 8e-4:
            p1.unlink()
            p2.unlink()
            return
        if np.linalg.norm(register(img, img2, 1)) > (1.5 / 0.325):
            p1.unlink()
            p2.unlink()
        return 


    _ = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(remove_bad_tiles)(
        ih, ii
    ) for ih, ii in zip(img_he_files, img_if_files))

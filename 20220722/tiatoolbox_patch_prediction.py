# 
# Ref -
# https://tia-toolbox.readthedocs.io/en/latest/_notebooks/05-patch-prediction.html
#
# workaround branch for in-house H&E images -
# https://github.com/Yu-AnChen/tiatoolbox/tree/d5f00215a23e6896618b8469f7e1b28743d88554
#
import palom
import tiatoolbox
import numpy as np
import skimage.transform

from tiatoolbox.models.engine.patch_predictor import PatchPredictor
from tiatoolbox.models.engine.patch_predictor import IOPatchPredictorConfig
from tiatoolbox.models.dataset.classification import WSIPatchDataset


def grid_shape(coors, patch_length=224):
    return grid_idx(coors, patch_length).max(axis=0) + 1


def grid_idx(coors, patch_length=224):
    index = np.fliplr(
        np.array(coors)[:, :2] /
        patch_length
    ).astype(int)
    return index


def construct_dataset(slide_path, do_mask=False):

    slide = tiatoolbox.wsicore.wsireader.TIFFWSIReader(
        slide_path,
         mpp=[MPP_TEST]*2,
         power=20,
         axes='SYX'
    )

    dataset = WSIPatchDataset(
        slide,
        mode='wsi',
        patch_input_shape=wsi_ioconfig.patch_input_shape,
        stride_shape=wsi_ioconfig.stride_shape,
        resolution=wsi_ioconfig.input_resolutions[0]["resolution"],
        units=wsi_ioconfig.input_resolutions[0]["units"],
        auto_get_mask=False,
    )
    g_shape = grid_shape(dataset.inputs)
    dataset.patch_idx = np.arange(np.multiply(*g_shape))
    dataset.grid_shape = g_shape

    if do_mask:
        
        c1r = palom.reader.OmePyramidReader(slide_path)
        level = -1 if len(c1r.pyramid) < 5 else 4
        mask = palom.img_util.entropy_mask(c1r.pyramid[level][1])
        mask = skimage.transform.resize(mask.astype(float), g_shape, order=3) > 0.25

        dataset.inputs = dataset.inputs[mask.flatten()]
        dataset.patch_idx = dataset.patch_idx[mask.flatten()]

    return dataset


predictor = PatchPredictor(
    pretrained_model='densenet161-kather100k', batch_size=32,
    num_loader_workers=0
)

# apply gamma correction before inference
import skimage.exposure
import torchvision.transforms

def preproc_func(img):
    return torchvision.transforms.ToTensor()(
        skimage.exposure.adjust_gamma(img, 2.2)
    ).permute(1, 2, 0)
# register preprocessing function
predictor.model.preproc_func = preproc_func

# inference configurations
ON_GPU = True

MPP_TEST = 0.325
MPP_TRAINING = 0.5
PATCH_WIDTH = int(224 * (0.5/MPP_TRAINING))

wsi_ioconfig = IOPatchPredictorConfig(
    # this is referring to the resolution of the training data
    # if the resolution here doesn't match the input wsi, the
    # input wsi will be resized
    input_resolutions=[{'units': 'mpp', 'resolution': MPP_TRAINING}],
    patch_input_shape=[PATCH_WIDTH, PATCH_WIDTH],
    stride_shape=[PATCH_WIDTH, PATCH_WIDTH]
)

# create dataset from WSI (H&E)
dataset = construct_dataset(
    r"X:\crc-scans\histowiz scans\Batch1\registered\tia-LSP10388.ome.tif",
    do_mask=True
)

# inference
p_results = predictor._predict_engine(
    dataset,
    return_labels=False,
    return_probabilities=True,
    return_coordinates=True,
    on_gpu=ON_GPU,
)

p_results['grid_shape'] = dataset.grid_shape

# visualize results
import matplotlib.pyplot as plt

img = np.zeros(dataset.grid_shape, np.uint8).flatten()
img[dataset.patch_idx] = p_results['predictions']
plt.figure()
plt.imshow(img, vmin=0, vmax=9, cmap='tab10')

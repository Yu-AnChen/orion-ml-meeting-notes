import skimage.metrics
import tifffile
import pathlib
import matplotlib.pyplot as plt


def set_matplotlib_font():
    font_families = matplotlib.rcParams['font.sans-serif']
    if font_families[0] != 'Arial':
        font_families.insert(0, 'Arial')
    matplotlib.rcParams['pdf.fonttype'] = 42


def save_all_fig(figs=None, dpi=144, label=None, close=True, out_dir=None):
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        num = fig.number
        title = '' if fig._suptitle is None else fig._suptitle.get_text()
        if label is None: label = ''
        title = label + title
        fig.suptitle(title)
        filename = f"fig {num}"
        if title:
            filename += f" - {title}"
        out_dir = pathlib.Path('.') if out_dir is None else pathlib.Path(out_dir)
        fig.savefig(out_dir / f"{filename.strip()}.png", bbox_inches='tight', dpi=dpi)
        if close:
            plt.close(num)


def compare(img1, img2, img_he=None):
    markers = ['Hoechst', 'AF', 'CD31', 'CD45', 'CD68', 'Blank', 'CD4', 'FOXP3', 'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163', 'E-Cadherin', 'PD-1', 'Ki-67', 'Pan-CK', 'SMA']
    fig = plt.figure(layout='constrained', figsize=(10, 4))
    _fig1, _fig2 = fig.subfigures(1, 2, wspace=0.02, width_ratios=[1.25, 5])

    if img_he is not None:
        axs1 = _fig1.subplots(2, 1)
        axs1[0].imshow(skimage.transform.resize(img_he, (256, 256)), alpha=0.5)
        axs1[0].imshow(np.dstack([img1[0], img1[0], np.zeros_like(img1[0])]), alpha=0.5)
        axs1[1].imshow(img_he)
        for ax in axs1.flat:
            ax.axis('off')
            ax.set_title('')

    axs2 = _fig2.subplots(4, 10)
    for i, ax, m in zip(img1, axs2.flat[::2], markers):
        ax.imshow(i, vmin=0, vmax=255)
        ax.set_title(m)
    for i, ax in zip(img2, axs2.flat[1::2]):
        ax.imshow(i, vmin=0, vmax=255)
    for i, j, ax in zip(img1, img2, axs2.flat[1::2]):
        ax.set_title(f"{skimage.metrics.normalized_mutual_information(i, j):.04f}")
    for ax in axs2.flat:
        ax.axis('off')
    return fig


gts = sorted(pathlib.Path('ground-truths').glob('*.tiff'))
preds = sorted(pathlib.Path('preds').glob('*.tiff'))
hes = sorted(pathlib.Path('img_he').glob('*.tif'))

img_gts = [np.moveaxis(tifffile.imread(p), 2, 0) for p in gts]
img_preds = [np.moveaxis(tifffile.imread(p), 2, 0) for p in preds]
img_he = [np.moveaxis(tifffile.imread(p), 0, 2) for p in hes]


set_matplotlib_font()
for i, j, k in zip(img_gts, img_preds, img_he):
    _ = compare(i, j, k)

save_all_fig(dpi=144)
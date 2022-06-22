import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from . import utils as blob_utils
from . import pvalue
from blobsar.logger import get_log

logger = get_log()


def plot_blobs(
    image=None,
    blobs=None,
    fig=None,
    ax=None,
    color="blue",
    blob_cmap=None,
    bbox=None,
    alpha=0.8,
    **kwargs,
):
    """Takes the blob results from find_blobs and overlays on image

    Can either make new figure of plot on top of existing axes.

    Returns:
        blobs
        ax
    """
    if fig and not ax:
        ax = fig.gca()
    if not ax:
        if bbox is not None:
            extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
        fig, ax = plt.subplots()
        ax_img = ax.imshow(image, extent=extent)
        fig.colorbar(ax_img)

    if not ax:
        ax = fig.gca()
    elif not fig:
        fig = ax.figure

    if blob_cmap:
        blob_cm = cm.get_cmap(blob_cmap, len(blobs))
    patches = []
    # Draw big blobs first to allow easier clicking on overlaps
    sorted_blobs = sorted(blobs, key=lambda b: b[2], reverse=True)
    for idx, blob in enumerate(sorted_blobs):
        if blob_cmap:
            color_pct = idx / len(blobs)
            color = blob_cm(color_pct)
        # print(f"Plotting  {blob[1], blob[0]}")
        c = plt.Circle(
            (blob[1], blob[0]),
            blob[2],
            color=color,
            fill=False,
            linewidth=2,
            alpha=alpha,
            clip_on=True,
            picker=True,
        )
        ax.add_patch(c)
        patches.append(c)

    # plt.draw()
    return blobs, ax


def plot_cropped_blob(image=None, blob=None, patch=None, crop_val=None, sigma=0):
    """Plot a 3d view of heighs of a blob along with its circle imshow view

    Args:
        image (ndarray): image in which blobs are detected
        blob: (row, col, radius, ...)
        patch (ndarray): optional: the sub-image from `crop_blob`, which is
            the area of `image` cropped around `blob`
        crop_val (float or nan): value to make all pixels outside sigma radius
            e.g. np.nan. if None, leaves the edges of bbox untouched
        sigma (float): if provided, smooth by a gaussian filter of size `sigma`

    Returns:
        matplotlib.Axes
    """
    if patch is None:
        patch = blob_utils.crop_blob(image, blob, crop_val=crop_val, sigma=sigma)
    elif sigma > 0:
        patch = blob_utils.gaussian_filter_nan(patch, sigma=sigma)
    ax = plot_surface(patch)
    return ax


def plot_surface(heights_grid, ax=None):
    """Makes default X, Y meshgrid to plot a surface of heights"""
    from mpl_toolkits.mplot3d import Axes3D
    rows, cols = heights_grid.shape
    xx = np.linspace(1, cols + 1, cols)
    yy = np.linspace(1, rows + 1, rows)
    X, Y = np.meshgrid(xx, yy)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection="3d")

    ax.plot_surface(X, Y, heights_grid)
    ax2 = fig.add_subplot(1, 2, 2)
    axim = ax2.imshow(heights_grid)
    fig.colorbar(axim, ax=ax2)
    return ax


def plot_kde(
    Z,
    extent,
    ax=None,
    resolution=None,
    cmap="Blues",
    title=None,
    grid=True,
    plot_log=False,
    vmax=None,
    vm_pct=100,
    figsize=(5, 5),
    show_colorbar=True,
    levels=15,
    ampcol=4,
    ylabel=None,
):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    # axim = ax.contourf(
    zplot = np.log(Z) if plot_log else Z
    if vmax is None:
        vmax = np.percentile(zplot, vm_pct)
    # axim = ax.imshow(
    xx = np.linspace(extent[0], extent[1], Z.shape[1])
    yy = np.linspace(extent[2], extent[3], Z.shape[0])
    axim = ax.contourf(
        xx,
        yy,
        # np.rot90(zplot),
        # zplot,
        np.rot90(zplot)[::-1],
        cmap=cmap,
        vmax=vmax,
        levels=levels,
        # origin="lower",
        # extent=extent,
        linewidths=0.5,
        extend="max",
        # alpha=.2,
    )
    ax.set_aspect("auto")
    if not ylabel:
        ylabel = r"$\bar{d}$ [cm]" if ampcol == 4 else r"${g}$"

    xlabel = r"$r$ [pixels]" if resolution is None else r"$r$ [km]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_colorbar:
        cbar = fig.colorbar(axim, ax=ax)
        label = "log PDF" if plot_log else "PDF"
        cbar.set_label(label)
    if title:
        ax.set_title(title)
    ax.grid(grid)


# ab = np.vstack((all_blobs, all_blobs2, all_blobs3))
# Z3, kernel3 = blobsar.plot.kde(np.abs(ab), vm_pct=90, bw_method=.3)


def hist2d(
    blobs,
    resolution=None,
    extent=None,
    density=True,
    use_abs=True,
    plot_log=False,
    logscale_x=False,
    ax=None,
    cmap="gist_earth_r",
    bins=50,
    title=None,
    grid=True,
    vm_pct=99,
    figsize=(5, 5),
):
    bb = np.abs(blobs.copy()) if use_abs else blobs.copy()
    if resolution is not None:
        print(f"Converting blob radii to km with {resolution = } meters")
        bb[:, 2] *= resolution / 1000

    if extent is None:
        print(f"Finding (radius, amplitude) extents of detected blobs")
        extent = pvalue.blob_extents(bb, 1.2)
    print(f"{extent = }")
    # [xmin, xmax, ymin, ymax] = extent
    # X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # positions = np.vstack([X.ravel(), Y.ravel()])

    # Only need the rad, Amp now
    bb = bb[:, -2:]
    Z, xedges, yedges = np.histogram2d(bb[:, 0], bb[:, 1], bins=bins, density=density)

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    zplot = np.log(Z) if plot_log else Z
    axim = ax.imshow(
        np.rot90(zplot),
        cmap=cmap,
        vmax=np.percentile(zplot, vm_pct),
        # origin="image",
        extent=extent,
    )
    if logscale_x:
        plt.xscale("log")
    ax.set_aspect("auto")
    fig.colorbar(axim, ax=ax)
    xlabel = r"$r$ [pixels]" if resolution is None else r"$r$ [km]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$A$ [cm]")
    if title:
        ax.set_title(title)
    ax.grid(grid)
    return Z, extent


def plot_top_detect(blobs, Z, extent, sar_date_list, dates):
    fig, axes = plt.subplots(2, 2)
    ztails = []
    for idx, (b, d) in enumerate(zip(blobs, dates)):
        ndates = sar_date_list.index(d) / 2
        zt = plot_detect(b, Z, extent, n_sar_dates=ndates, ax=axes.ravel()[idx])
        ztails.append(zt)

    return ztails


def plot_detect(Z, blob, extent, stacking_factor=[1], p0_factor=1, ax=None, title=None):
    if np.isscalar(stacking_factor):
        stacking_factor = [stacking_factor]
    if not ax:
        fig, ax = plt.subplots()

    ax.axvline(np.abs(blob[-1]), linestyle="-", color="k")
    for sf in stacking_factor:
        Avec, z_conditional, pvalue = pvalue.find_pvalue(
            Z, extent=extent, blob=blob, stacking_factor=sf, p0_factor=p0_factor
        )
        ax.plot(Avec, z_conditional, label=f"{sf}", lw=3)
        logger.info(f"For {sf} stacking factor, p = {pvalue:.3f}")
    ax.set_xlabel(r"$A$ [cm]")
    ax.set_ylabel(r"$pdf$")
    if title is None:
        ax.set_title(f"p = {pvalue:.3f}")
    ax.legend()
    plt.tight_layout()

    return pvalue


def filtmag_vs_mag(
    filt_threshs=np.linspace(0.1, 0.2, 5)[::-1], nbins=100, pval_thresh=0.05, r0_idx=10, plot=True,
):
    import scipy.stats as stats
    import pandas as pd

    barr = pvalue.load_all_blobs(".")
    df = pd.DataFrame(barr, columns=["row", "col", "r", "filt", "mag"])
    rlist = sorted(df.r.unique())
    r0 = rlist[r0_idx]

    dfabs = df[["r", "filt", "mag"]].abs()
    mags = dfabs[dfabs.r == r0].mag

    # bins = np.histogram_bin_edges(mags, bins=nbins)
    bins = np.linspace(0.8 * mags.min(), 1.2 * mags.max(), nbins)

    densities = []
    for fthresh in filt_threshs:

        data = dfabs[(dfabs.r == r0) & (dfabs.filt > fthresh)].mag
        if data.size <= 1:
            density = None
        else:
            density = stats.gaussian_kde(data)
        densities.append(density)
        # data.plot.hist( bins=bins, density=True, label=fthresh, alpha=0.5)
    if plot:
        fig, ax = plt.subplots()
        for fthresh, density in zip(filt_threshs, densities):
            ax.plot(bins, density(bins), label=f"{fthresh:.2f}")
        ax.legend()
        ax.set_xlabel("mag")

    # cdfs = [np.cumsum(d(bins) * np.diff(bins)[0]) for d in densities]
    cdfs = []
    for d in densities:
        # cdf = [d.integrate_box(0, high) for high in np.linspace(0.1, 1., 100)]
        if d is None:
            cdf = None
        else:
            cdf = np.array([d.integrate_box(0, b) for b in bins])
        cdfs.append(cdf)
    # cdfs = np.array(cdfs)

    mag_cutoffs = []
    for c in cdfs:
        if c is None:
            mag_cutoff = None
        else:
            idx95 = np.argwhere(c > (1 - pval_thresh))[0][0]
            mag_cutoff = bins[idx95]
        mag_cutoffs.append(mag_cutoff)

    return bins, densities, cdfs, mag_cutoffs

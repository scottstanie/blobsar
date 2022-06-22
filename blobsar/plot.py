import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.qhull import ConvexHull
from sklearn import manifold
from apertools import plotting
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
    plot_img=False,
    delete=False,
    extent=None,
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
    if plot_img or not ax:
        fig, ax, ax_img = plotting.plot_image(
            image, fig=fig, ax=ax, extent=extent, bbox=bbox, **kwargs
        )
        # ax_img = ax.imshow(image)
        # fig.colorbar(ax_img)

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

    remaining_blobs = blobs
    # plt.draw()
    if delete is False:
        # print("ok")
        # plt.show()
        return blobs, ax

    ax.blobs = sorted_blobs
    ax.picked_idx = None
    ax.picked_object = None
    ax.deleted_idxs = set()

    pick_handler = on_pick(sorted_blobs, patches)
    cid_pick = fig.canvas.mpl_connect("pick_event", pick_handler)
    cid_press = fig.canvas.mpl_connect("button_press_event", on_press)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    if ax.deleted_idxs:
        print("Deleted %s blobs" % len(ax.deleted_idxs))
        all_idx = range(len(blobs))
        remaining = list(set(all_idx) - set(ax.deleted_idxs))
        remaining_blobs = np.array(sorted_blobs)[remaining]
    else:
        remaining_blobs = blobs

    fig.canvas.mpl_disconnect(cid_pick)
    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_key)

    return remaining_blobs, ax


def on_pick(blobs, patches):
    def pick_event(event):
        """Store the index matching the clicked blob to delete"""
        ax = event.artist.axes
        for i, artist in enumerate(patches):
            if event.artist == artist:
                ax.picked_idx = i
                ax.picked_object = artist  # Also save circle Artist to remove

        print("Selected blob: %s" % str(blobs[ax.picked_idx]))

    return pick_event


def on_press(event):
    "on button press we will see if the mouse is over us and store some data"
    # print("on press event", event)
    # You can either double click or right click to unselect
    if event.button != 3 and not event.dblclick:
        return

    ax = event.inaxes
    if ax:
        print("Unselecting blob")
        ax.picked_idx = None
        ax.picked_object = None


def on_key(event):
    """
    Function to be bound to the key press event
    If the key pressed is delete and there is a picked object,
    remove that object from the canvas
    """
    if event.key == "delete":
        ax = event.inaxes
        if ax is not None and ax.picked_object:
            cur_blob = ax.blobs[ax.picked_idx]
            print("Deleting blob %s" % str(cur_blob))
            ax.deleted_idxs.add(ax.picked_idx)
            ax.picked_idx = None

            ax.picked_object.remove()
            ax.picked_object = None
            ax.figure.canvas.draw()


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


# TODO: export this? seems useful
def plot_surface(heights_grid, ax=None):
    """Makes default X, Y meshgrid to plot a surface of heights"""
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


def scatter_blobs(blobs, image=None, axes=None, color="b", label=None):
    if axes is None:
        fig, axes = plt.subplots(1, 3)
    else:
        fig = axes[0].get_figure()

    if blobs.shape[1] < 6:
        blobs = blob_utils.append_stats(blobs, image)

    print("Taking abs value of blobs")
    blobs = np.abs(blobs)

    # Size vs amplitude
    sizes = blobs[:, 2]
    mags = blobs[:, 3]
    vars_ = blobs[:, 4]
    ptps = blobs[:, 5]

    axes[0].scatter(sizes, mags, c=color, label=label)
    axes[0].set_xlabel("Size")
    axes[0].set_ylabel("Magnitude")
    if label:
        axes[0].legend()

    axes[1].scatter(sizes, vars_, c=color, label=label)
    axes[1].set_xlabel("Size")
    axes[1].set_ylabel("variance")

    axes[2].scatter(sizes, ptps, c=color, label=label)
    axes[2].set_xlabel("Size")
    axes[2].set_ylabel("peak-to-peak")
    return fig, axes


def scatter_blobs_3d(blobs, image=None, ax=None, color="b", label=None, blob_img=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        fig = ax.get_figure()

    if blobs.shape[1] < 6:
        blobs = blob_utils.append_stats(blobs, image)

    if blob_img is not None:
        # Length of radii in km
        sizes = blob_img.pixel_to_km(blobs[:, 2])
    else:
        sizes = blobs[:, 2]
    mags = blobs[:, 3]
    vars_ = blobs[:, 4]
    ax.scatter(sizes, mags, vars_, c=color, label=label)
    ax.set_title("Size, mag, var of blobs")
    ax.set_xlabel("size")
    ax.set_ylabel("magniture")
    ax.set_zlabel("variance")
    return fig, ax


def plot_hull(regions=None, hull=None, ax=None, linecolor="k-"):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if hull is None:
        hull = ConvexHull(regions)
    for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], linecolor)


def plot_bbox(bbox, ax=None, linecolor="k-", cv_format=False):
    for c in blob_utils.bbox_to_coords(bbox, cv_format=cv_format):
        print(c)
        ax.plot(c[0], c[1], "rx", markersize=6)


def plot_regions(regions, ax=None, linecolor="k-"):
    for shape in blob_utils.regions_to_shapes(regions):
        xx, yy = shape.convex_hull.exterior.xy
        ax.plot(xx, yy, linecolor)


def plot_tsne(X, y_idxs=None, n_components=2, perplexities=(5,), colors=("r", "g")):
    fig, axes = plt.subplots(1, len(perplexities))
    Y_list = []
    for pidx, p in enumerate(perplexities):
        tsne = manifold.TSNE(n_components=n_components, perplexity=p, init="random")
        Y = tsne.fit_transform(X)
        Y_list.append(Y)

        ax = axes.ravel()[pidx]
        if y_idxs is not None:
            for j, idxs in enumerate(y_idxs):
                ax.scatter(Y[idxs, 0], Y[idxs, 1], c=colors[j])
        else:
            ax.scatter(Y[:, 0], Y[:, 1])
        ax.set_title("perplexity=%s" % p)

    return Y_list


def plot_scores(score_arr, nrows=1, y_idxs=None, titles=None):
    n_scores = score_arr.shape[1]
    ncols = np.ceil(n_scores / nrows).astype(int)
    fix, axes = plt.subplots(nrows, ncols, squeeze=False)

    if y_idxs is None:
        y_idxs = np.arange(len(score_arr))
    if titles is None:
        titles = np.range(n_scores).astype(int)

    for idx in range(n_scores):
        ax = axes.ravel()[idx]
        for y_idx, yy in enumerate(y_idxs):
            ax.hist(score_arr[yy, idx], bins=50, alpha=0.5, label=y_idx)
        ax.set_title(titles[idx])
        ax.legend()


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


if __name__ == "__main__":
    # npz = np.load('patches/image_1.npz')
    # image = npz['image']
    # real_blobs = npz['real_blobs']
    # plot_blobs(image=image, blobs=real_blobs)
    import sys
    import json
    import geog
    import shapely.geometry

    if len(sys.argv) < 4:
        print("python %s lat lon radius(in km)" % sys.argv[0])
    else:
        print("python %s lat lon radius(in km)" % sys.argv[0])

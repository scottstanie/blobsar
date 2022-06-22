"""Module for finding blobs in deformation maps
"""
import numpy as np

from . import utils as blob_utils
from .logger import get_log

logger = get_log()
BLOB_KWARG_DEFAULTS = {"threshold": 1, "min_sigma": 3, "max_sigma": 40}


# TODO: clean up redundancies in this function
def find_blobs(
    image,
    positive=True,
    negative=True,
    threshold=0.5,
    mag_threshold=None,
    min_sigma=3,
    max_sigma=60,
    num_sigma=20,
    sigma_bins=1,
    min_km=None,
    max_km=None,
    resolution=None,
    prune_edges=True,
    radius_pct=0.75,
    border_size=2,
    log_scale=True,
    sigma_ratio=1.4,
    verbose=0,
    **kwargs,
):
    """Find blob features within an image

    Args:
        image (ndarray): image containing blobs
        positive (bool): default True: if True, searches for positive (light, uplift)
            blobs within image
        negative (bool): default True: if True, finds dark, subsidence blobs
        mag_threshold (float): absolute value in the image blob must exceed
            Should be positive number even if negative=True (since image is inverted)
        threshold (float): response threshold passed to the blob finding function
        min_sigma (int): minimum pixel size to check for blobs
        max_sigma (int): max pixel size to check for blobs
        num_sigma : int, optional: number of intermediate values of filter size to use
            blobs that are within the same bin (to keep very large and nested
            small blobs). int passed will divide range evenly
        sigma_bins : int or array-like of edges: Will only prune overlapping
        min_km (float): minimum blob size in km
            assumes image is a LatlonImage, and will use the dem rsc info
            to convert km to pixels
        max_km (float): maximum blob size in km
        prune_edges (bool):  will look for "ghost blobs" near strong extrema to remove,
        radius_pct (float): (default 1.) Only use `radius_pct` of the middle of the blob
            when getting the "magnitude" of blob
        border_size (int): Blobs with centers within `border_size` pixels of
            image borders will be discarded
        bowl_score (float): if > 0, will compute the shape score and only accept
            blobs that with higher shape_index than `bowl_score`
        log_scale : bool, optional
            If set intermediate values of standard deviations are interpolated
            using a logarithmic scale. If not, linear

    Returns:
        blobs: ndarray: rows are blobs with values: [(r, c, radius, mag)], where
            r = row num of center, c is column, radius is sqrt(2)*sigma for
            the std. dev. of Gaussian that detected blob), and
            ``mag`` is the (Gaussian weighted) maximum
            amplitude of the image contained within the blob.
        sigma_list: ndarray
            array of sigma values used to filter scale space

    Notes:
        kwargs are passed to the blob_log function (such as overlap).
        See reference for full list

    Reference:
    [1] http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html
    """
    from . import skblob

    def _dist_to_sigma(dist_km, resolution):
        if resolution is None:
            raise ValueError("Need `resolution` passed if using `min_km/max_km`")
        radius_pix = dist_km * 1000 / resolution
        sigma = radius_pix / np.sqrt(2)
        logger.info(f"Converted {dist_km} to sigma={sigma} pixels")
        return sigma

    # some skimage funcs fail for float32 when unnormalized [0,1]
    image = image.astype("float64")
    if min_km is not None:
        logger.info(f"Using {min_km = }")
        min_sigma = _dist_to_sigma(min_km, resolution)
    if max_km is not None:
        logger.info(f"Using {max_km = }")
        max_sigma = _dist_to_sigma(max_km, resolution)

    sigma_list = skblob.create_sigma_list(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        log_scale=log_scale,
        sigma_ratio=sigma_ratio,
    )

    blobs = np.empty((0, 5))

    image_cube = skblob.create_gl_cube(image, sigma_list=sigma_list)

    if positive:
        # logger.info('bpos')
        blobs_pos = skblob.blob_log(
            image=image,
            threshold=threshold,
            image_cube=image_cube,
            sigma_list=sigma_list,
            sigma_bins=sigma_bins,
            prune_edges=prune_edges,
            border_size=border_size,
            positive=True,
            **kwargs,
        )
        if int(verbose) > 1:
            logger.info(blobs_pos)
        # Append mags as a column and sort by it
        # TODO: FIX vvv
        if blobs_pos.size:
            blobs_with_mags = blob_utils.sort_blobs_by_val(
                blobs_pos, image, positive=True, radius_pct=radius_pct
            )
        else:
            blobs_with_mags = np.empty((0, 5))
        # logger.info(blobs_with_mags)
        if mag_threshold is not None:
            blobs_with_mags = blobs_with_mags[blobs_with_mags[:, -1] >= mag_threshold]
        blobs = np.vstack((blobs, blobs_with_mags))
        if int(verbose) > 0:
            logger.info(f"Found {len(blobs_with_mags)} positive blobs")

    if negative:
        # logger.info('bneg')
        blobs_neg = skblob.blob_log(
            image=image,
            threshold=threshold,
            image_cube=-1 * image_cube,
            sigma_list=sigma_list,
            sigma_bins=sigma_bins,
            prune_edges=prune_edges,
            border_size=border_size,
            positive=False,
            **kwargs,
        )
        if blobs_neg.size:
            blobs_with_mags = blob_utils.sort_blobs_by_val(
                blobs_neg, image, positive=False, radius_pct=radius_pct
            )
        else:
            blobs_with_mags = np.empty((0, 5))
        # logger.info(blobs_with_mags)
        if mag_threshold is not None:
            blobs_with_mags = blobs_with_mags[
                -1 * blobs_with_mags[:, -1] >= mag_threshold
            ]
        blobs = np.vstack((blobs, blobs_with_mags))
        if int(verbose) > 0:
            logger.info(f"Found {len(blobs_with_mags)} negative blobs")

    return blobs, sigma_list

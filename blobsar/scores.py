"""Module of scoring functions for blob patches"""
import numpy as np
import multiprocessing
from math import sqrt
from scipy import ndimage as ndi
from . import utils as blob_utils
from .skblob import shape_index


def shape_index_stat(
    patch, accum_func, sigma=None, sigma_scale=None, patch_size="auto"
):
    """Finds some state about the shape_index of a patch

    Args:
        patch: ndarray, pixel box around blob center
        sigma (float): specific size gaussian to smooth by
        sigma_scale (float): number > 1 to divide the sigma
            of the patch by
        patch_size (int or str): choices are 'auto', 'full', or int
            'auto' takes the size of the blob and divides
            by 5 for a proportional patch
            'full' takes patch size of sigma
            int will use that patch size

    Returns:
        float: score of shape_index within patch size
    """
    # If they don't specify one sigma, use patch size
    if sigma is None:
        sigma = blob_utils.sigma_from_blob(patch=patch)
        # print('sfp', sigma)
    # if they want to scale down the default (which smooths proportional to the blob sigma)
    if sigma_scale:
        sigma /= sigma_scale

    if patch_size == "auto":
        # 2 * because patch size excepts height, not radius
        psize = int(2 * max(3, sigma / 5))
    elif patch_size == "full":
        psize = patch.shape[0]
    else:
        psize = patch_size
    if not isinstance(psize, int):
        raise ValueError("patch_size must be int")
    # print('psize', psize)

    # For all these functions, grab the image of shape indexes to use
    shape_index_arr = shape_index(patch, sigma=sigma)
    return blob_utils.get_center_value(
        shape_index_arr, patch_size=psize, accum_func=accum_func
    )


def shape_index_center(patch):
    """Finds the mean shape_index of a 3x3 patch around center pixel
    sigma=proportional to the patch size computed with sigma_from_patch"""
    return shape_index_stat(patch, np.mean, sigma=None, patch_size=3)


def shape_index_center_sigma1(patch):
    """Finds the mean shape_index of a 3x3 patch around center pixel, sigma=1"""
    return shape_index_stat(patch, np.mean, sigma=1, patch_size=3)


def shape_index_center_sigma3(patch):
    """Finds the mean shape_index of a 3x3 patch around center pixel, sigma=3"""
    return shape_index_stat(patch, np.mean, sigma=3, patch_size=3)


def shape_index_center_min_sigma3(patch):
    """Finds the min shape_index of a 3x3 patch around center pixel, sigma=3"""
    return shape_index_stat(patch, lambda x: np.min(np.mean(x)), sigma=3, patch_size=3)


def shape_index_center_min_sigma1(patch):
    """Finds the min shape_index of a 3x3 patch around center pixel, sigma=3"""
    return shape_index_stat(patch, lambda x: np.min(np.mean(x)), sigma=1, patch_size=1)


def shape_index_variance_patch3_sigma3(patch):
    """Smooth by a small sigma=3, look in a patch=3 at variance"""
    return shape_index_stat(patch, np.var, sigma=3, patch_size=3)


def shape_index_variance_patch_full_sigma3(patch):
    """Smooth by a small sigma=3, look at entire patch for variance"""
    return shape_index_stat(patch, np.var, sigma=3, patch_size="full")


def shape_index_variance_patch_full(patch):
    """Smooth over a large sigma equal to blob sigma, take variance over all patch"""
    return shape_index_stat(patch, np.var, sigma=None, patch_size="full")


def shape_index_ptp_patch3(patch):
    """Check peak-to-peak in patch=3, smoothing with sigma proportional"""
    return shape_index_stat(patch, np.ptp, sigma=None, patch_size=3)


def shape_index_ptp_patch_full(patch):
    """Look in a total patch for large changes in the shape index peak-to-peak"""
    return shape_index_stat(patch, np.ptp, sigma=None, patch_size="full")


def max_gradient(patch, sigma=0.5):
    p = blob_utils.gaussian_filter_nan(patch, sigma=sigma)
    imy = np.abs(ndi.sobel(patch, axis=0, mode="nearest"))
    imx = np.abs(ndi.sobel(patch, axis=1, mode="nearest"))
    return max(np.max(imx), np.max(imy))


def max_gradient_sigma3(patch):
    return max_gradient(patch, sigma=3)


FUNC_LIST = [
    shape_index_center,
    shape_index_center_sigma1,
    shape_index_center_sigma3,
    shape_index_center_min_sigma3,
    shape_index_center_min_sigma1,
    shape_index_variance_patch3_sigma3,
    shape_index_variance_patch_full_sigma3,
    shape_index_variance_patch_full,
    shape_index_ptp_patch3,
    shape_index_ptp_patch_full,
    max_gradient,
    max_gradient_sigma3,
]

FUNC_LIST_NAMES = [f.__name__ for f in FUNC_LIST]


def analyze_patches(patch_list, funcs=FUNC_LIST, *args, **kwargs):
    """Get scores from functions on a series of patches

    Runs each function in `funcs` over each `patch` to get stats on it
    Each function must have a signature func(patch, *args, **kwargs),
        and return a single float number
    Args:
        patch_list:
        funcs:
        *args:
        **kwargs:

    Returns:
        ndarray: size (p, N) where p = num patches, N = len(funcs)
            rows are scores on one patch
    """
    results = []
    for patch in patch_list:
        results.append([func(patch, *args, **kwargs) for func in funcs])
    return np.array(results)



def _corner_score(image, sigma=1):
    """wrapper for corner_harris to normalize for multiprocessing"""
    from skimage import feature
    return sigma ** 2 * feature.corner_harris(image, sigma=sigma)


def compute_blob_scores(
    image, sigma_list, score="shape", find_peaks=True, gamma=1.4, threshold_rel=0.1
):
    """Computes blob score on image at each sigma, finds peaks

    Possible scores:
        shape index: measure of "bowl"ness at a point for a given sigma
            using Hessian eigenvalues. see skimage.feature.shape_index
        harris corner: finds "cornerness" using autocorrelation eigenvalues

    Args:
        image (ndarray): input image to compute corner harris reponse
        sigma_list (array-like): output of create_sigma_list
        score (str): choices: 'shape', 'harris' (or 'corner'): which function
            to use to score each blob layer
        find_peaks (bool): if true, also runs peak_local_max on each layer to find
            the local peaks of scores
        gamma (float): adjustment from sigma in LoG to harris (if using Harris)
            The Gaussian kernel for the LoG scale space (t) is smaller
            than the window used to pre-smooth and find the Harris response (s),
            where s = t * gamma**2
        threshold_rel (float): passed to find peaks. Using relative since
            smaller sigmas for corner_harris have higher

    Returns:
        peaks: output of peak_local_max on stack of corner responses
        score_imgs: the score response at each level

    TODO: see if doing corner_harris * s**2 to normalize, and using threshold_abs
        works better than threshold_rel

    Sources:
        https://en.wikipedia.org/wiki/Corner_detection#The_multi-scale_Harris_operator
        Koenderink, J. J. & van Doorn, A. J.,
           "Surface shape and curvature scales",
           Image and Vision Computing, 1992, 10, 557-564.
           :DOI:`10.1016/0262-8856(92)90076-F`
    """
    from . import skblob

    valid_scores = ("corner", "harris", "shape")
    if score not in valid_scores:
        raise ValueError("'score' must be one of: %s" % str(valid_scores))

    pool = multiprocessing.Pool()
    jobs = []
    for s in sigma_list:
        if score == "corner" or score == "harris":
            jobs.append(pool.apply_async(_corner_score, (image,), {"sigma": s * gamma}))
        elif score == "shape":
            jobs.append(
                pool.apply_async(
                    skblob.shape_index, (image,), {"sigma": s, "mode": "nearest"}
                )
            )

    score_imgs = [result.get() for result, s in zip(jobs, sigma_list)]

    if find_peaks:
        jobs = []
        for layer in score_imgs:
            jobs.append(
                pool.apply_async(
                    skblob.peak_local_max, (layer,), {"threshold_rel": threshold_rel}
                )
            )
        peaks = [result.get() for result in jobs]
    else:
        peaks = None

    return score_imgs, peaks


# # TODO: use the scores.py module instead of repeat here
# def get_blob_bowl_score(image, blob, sigma=None, patch_size=3):
#     patch = blob_utils.crop_blob(image, blob, crop_val=None)  # Don't crop with nans
#     shape_vals = skblob.shape_index(patch, sigma=sigma, mode="nearest")
#     return blob_utils.get_center_value(shape_vals, patch_size=patch_size)


def find_blobs_with_bowl_scores(image, blobs=None, score_cutoff=0.7, **kwargs):
    """Takes the list of blobs found from find_blobs, check for high shape score

    Computes a shape_index, finds blobs that have
    a |shape_index| > 5/8 (which indicate a bowl shape either up or down.
    Blobs with no corner peak found are discarded (they are valleys or ridges)

    Args:
        image (ndarray): input image to compute corners on
        blobs (ndarray): rows are blobs with values: [(r, c, s, ...)]
        score_cutoff (float): magnitude of shape index to approve of bowl blob
            Default is .7: from [1], Slightly more "bowl"ish, cutoff at 7/8,
            than "trough", at 5/8. Shapes at 5/8 still look "rut" ish, like
            a valley
        patch_size (int):
        kwargs: passed to find_blobs if `blobs` not passed as argument

    Returns:
        ndarray: like blobs, with some rows deleted that contained no corners

    References:
        [1] Koenderink, J. J. & van Doorn, A. J.,
           "Surface shape and curvature scales",
           Image and Vision Computing, 1992, 10, 557-564.
           :DOI:`10.1016/0262-8856(92)90076-F`
    """
    if blobs is None:
        blobs, sigma_list = find_blobs(image, **kwargs)

    # OLD: Find peaks for every sigma in sigma_list
    # score_images, _ = compute_blob_scores(image, sigma_list, find_peaks=False)

    # sigma_idxs = blob_utils.find_sigma_idxs(blobs, sigma_list)
    # Note: using smaller sigma than blob size seems to work better in bowl scoring
    # sigma_arr = sigma_list[sigma_idxs]
    # sigma_arr = np.clip(sigma_list[sigma_idxs] / 10, 2, None)
    # sigma_arr = 2 * np.ones(len(blobs))

    # import ipdb
    # ipdb.set_trace()
    out_blobs = []
    # for b, sigma in zip(blobs, sigma_arr):
    # for blob, sigma_idx in zip(blobs, sigma_idxs):
    # OLD WAY: compute scores, then crop. THIS is DIFFERENT than crop, then score for bowlness
    # # Get the peaks that correspond to the current sigma level
    # cur_scores = score_images[sigma_idx]
    # # Only examine blob area
    # blob_scores = blob_utils.crop_blob(cur_scores, blob, crop_val=None)
    # center_score = blob_utils.get_center_value(blob_scores)
    for b in blobs:
        sigma = blob_utils.sigma_from_blob(blob=b)
        center_score = get_blob_bowl_score(image, b, sigma=sigma)
        if np.abs(center_score) >= score_cutoff:
            out_blobs.append(b)
        else:
            print("removing blob: %s, score: %s" % (str(b.astype(int)), center_score))

    return np.array(out_blobs)


def find_blobs_with_harris_peaks(
    image, blobs=None, sigma_list=None, gamma=1.4, threshold_rel=0.1, **kwargs
):
    """Takes the list of blobs found from find_blobs, check for high cornerness

    Computes a harris corner response at each level gamma*sigma_list, finds
    peaks, then checks if blobs at that sigma level have some corner inside.
    Blobs with no corner peak found are discarded (they are edges or ghost
    blobs found at the ring of sharp real blobs)

    Args:
        image (ndarray): input image to compute corners on
        blobs (ndarray): rows are blobs with values: [(r, c, s, ...)]
        sigma_list (array-like): output of create_sigma_list
        gamma (float): adjustment from sigma in LoG to harris
        threshold_rel (float): passed to find peaks to threshold real peaks.
        kwargs: passed to find_blobs if `blobs` not passed as argument

    Returns:
        ndarray: like blobs, with some rows deleted that contained no corners
    """
    if blobs is None:
        blobs, sigma_list = blobsar.find_blobs(image, **kwargs)

    # Find peaks for every sigma in sigma_list
    _, corner_peaks = compute_blob_scores(
        image, sigma_list, gamma=gamma, threshold_rel=threshold_rel
    )

    sigma_idxs = blob_utils.find_sigma_idxs(blobs, sigma_list)
    # import ipdb
    # ipdb.set_trace()
    out_blobs = []
    for blob, sigma_idx in zip(blobs, sigma_idxs):
        # Get the bowl_score that correspond to the current sigma level
        cur_peaks = corner_peaks[sigma_idx]
        # Only examine blob area
        blob_mask = blob_utils.indexes_within_circle(blob=blob, mask_shape=image.shape)
        corners_contained_in_mask = blob_mask[cur_peaks[:, 0], cur_peaks[:, 1]]
        # corners_contained_in_mask = blob_mask[cur_peaks[:, 1], cur_peaks[:, 0]]
        if any(corners_contained_in_mask):
            out_blobs.append(blob)

    return np.array(out_blobs)

"""utils.py: Functions for finding blobs in deformation maps
"""
import collections
import itertools
import apertools.latlon
import numpy as np
import skimage
from scipy.ndimage import gaussian_filter

# from scipy.spatial.qhull import ConvexHull
# from shapely.geometry import Point, MultiPoint, Polygon, box
import geog
import shapely.geometry
import geojson


def get_center_value(img, patch_size=1, accum_func=np.mean):
    """Find center of image, taking reducing around `patch_size` pixels

    Args:
        img (ndarray): 2D image to get center value
        patch_size (int): number of pixels to look around center for
            this will be the width/height. e.g. patch_size=3 means (3,3)
        accum_func (numpy function): default = np.mean.
            Reduces pixels in patch_size into one number

    Returns:

    """
    rows, cols = img.shape
    rcent = rows // 2
    ccent = cols // 2
    p = patch_size // 2
    return accum_func(img[rcent - p : rcent + p + 1, ccent - p : ccent + p + 1])


def indexes_within_circle(mask_shape=None, center=None, radius=None, blob=None):
    """Get a mask of indexes within a circle

    Args:
        center (tuple[float, float]): row, column of center of circle
        radius (float): radius of circle
        mask_shape (tuple[int, int]) rows, cols to make mask for entire image
        blob (tuple[float, float, float]): row, col, radius of blob
            This option is instead of using `center` and `radius`
    Returns:
       np.array[bool]: boolean mask of size `mask_shape`
    """
    if mask_shape is None:
        raise ValueError("Need mask_shape to determine output array size")
    height, width = mask_shape
    if blob is not None:
        cy, cx, radius = blob[:3]
    elif center is not None:
        cy, cx = center
    if radius is None:
        raise ValueError("Need radius if not using `blob` input")
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return dist_from_center <= radius


def get_blob_stats(blobs, image, center_only=False, radius_pct=1.0, accum_func=np.max):
    """Find statistics about image values within each blob

    Checks all pixels within the radius of the blob, and runs some
    numpy function `accum_func` on these values

    Args:
        blobs (ndarray): 2D, entries [row, col, radius, ...], from find_blobs
        image (ndarray): 2D image where blobs were found
        center_only (bool): (default False) Only get the value of the center pixel of the blob
        radius_pct (float): (default 1.) Only use `radius_pct` of the middle of the blob
        accum_func (bool): (default np.max) Function to run on all pixels
            within blob to accumulate to one value

    Returns:
        ndarray: length = N, number of blobs, each value is the max of the image
        within the blob radius.
        If all_pixels = True, each entry is a list of pixel values
    """
    if center_only:
        coords = blobs[:, :2].astype(int)
        return image[coords[:, 0], coords[:, 1]]

    blobs_adjusted = blobs.copy()
    if radius_pct != 1:
        blobs_adjusted[:, 2] *= radius_pct

    # blob: [row, col, radius, [possibly mag]]
    masks = [
        indexes_within_circle(blob=blob, mask_shape=image.shape)
        for blob in blobs_adjusted
    ]
    return np.stack([accum_func(image[mask]) for mask in masks])


def mask_border(mask):
    """Takes a mask (bool ndarray), finds ts bounding box"""
    rows, cols = np.where(mask)
    return np.min(rows), np.max(rows), np.min(cols), np.max(cols)


def gauss_weighted_func(patch, sigma=None, func=np.max):
    """Get the max of a patch after weighting by a gaussian (not filter, just multiply)"""
    if sigma is None:
        sigma = patch.shape[0] / 2
    # raise ValueError("HI")
    return func(gaussian_kernel(patch.shape, sigma=sigma) * patch)


def get_blob_stats_weighted(blobs, image, accum_func=gauss_weighted_func):
    """Find statistics about image values within each blob

    Checks all pixels within the radius of the blob, and runs some
    numpy function `accum_func` on these values

    Args:
        blobs (ndarray): 2D, entries [row, col, radius, ...], from find_blobs
        image (ndarray): 2D image where blobs were found
        center_only (bool): (default False) Only get the value of the center pixel of the blob
        radius_pct (float): (default 1.) Only use `radius_pct` of the middle of the blob
        accum_func (bool): (default gauss_weighted_max) Function to run on all pixels
            within blob to accumulate to one value

    Returns:
        ndarray: length = N, number of blobs, each value is the max of the image
        within the blob radius.
        If all_pixels = True, each entry is a list of pixel values
    """
    # blobs_adjusted = blobs.copy()
    blobs_adjusted = blobs
    # if radius_pct != 1:
    # blobs_adjusted[:, 2] *= radius_pct

    # blob: [row, col, radius, [possibly mag]]
    patches = [
        patch_within_square(image, blob=blob, mask_shape=image.shape)
        for blob in blobs_adjusted
    ]
    return np.stack([accum_func(patch) for patch in patches])


def patch_within_square(image, mask_shape=None, blob=None):
    """Get a mask of indexes within a square centered at blob."""
    if mask_shape is None:
        raise ValueError("Need mask_shape to determine output array size")
    height, width = mask_shape
    if blob is not None:
        row, col, radius = blob[:3]
    Y, X = np.ogrid[:height, :width]
    min_row = max(0, int(row - radius))
    max_row = int(row + radius)
    min_col = max(0, int(col - radius))
    max_col = int(col + radius)
    return image[min_row : max_row + 1, min_col : max_col + 1]


def gaussian_kernel(shape, sigma=None, pdf=False):
    """Create a Gaussian kernal with std. dev `sigma` to weight a patch """
    ny, nx = shape
    if sigma is None:
        sigma = min(nx, ny) / 4
    y = (np.arange(ny) - ny // 2).reshape((-1, 1))
    x = (np.arange(nx) - nx // 2).reshape((1, -1))
    C = 1 / (2 * np.pi * sigma ** 2) if pdf else 1
    return C * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def crop_image_to_mask(image, mask, crop_val=None, sigma=0):
    """Returns only part of `image` within the bounding box of `mask`

    Args:
        image (ndarray): image to crop
        mask: boolean ndarray same size as image from indexes_within_circle
            if provided (e.g. with np.nan), area around blob circle gets
            masked and set to this value. Otherwise, a square patch is used
        crop_val (float or nan): value to make all pixels outside sigma radius
            default=np.nan. if None, leaves the edges of bbox untouched
        sigma (float): if provided, smooth by a gaussian filter of size `sigma`
    """
    masked_out = image.copy()
    if sigma > 0:
        print("filtering")
        masked_out = gaussian_filter(masked_out, sigma=sigma)
    if crop_val is not None:
        masked_out[~mask] = crop_val
    # Now find square border of blob circle and crop
    min_row, max_row, min_col, max_col = mask_border(mask)
    return masked_out[min_row : max_row + 1, min_col : max_col + 1]


def crop_blob(image, blob, crop_val=None, sigma=0):
    """Crops an image to the box around a blob with nans outside blob area
    Args:
        image:
        blob: (row, col, radius, ...)
        crop_val (float or nan): value to make all pixels outside sigma radius
            e.g with p.nan. default=None, leaves the edges of bbox untouched
        sigma (float): if provided, smooth by a gaussian filter of size `sigma`

    Returns:
        ndarray: size = (2r, 2r), r = radius of blob
    """
    mask = indexes_within_circle(blob=blob, mask_shape=image.shape)
    return crop_image_to_mask(image, mask, crop_val=crop_val, sigma=sigma)


def append_stats(blobs, image, stat_funcs=(np.var, np.ptp), center_only=False):
    """Append columns based on the statistic functions in stats

    Default: adds the variance and peak-to-peak within blob

    Args:
        center_only (bool, or array-list[bool]): pass to get_blob_stats
            Can either do 1 true false for all stat_funcs, or an iterable
            of matching size
    """
    new_blobs = blobs.copy()
    if isinstance(center_only, collections.Iterable):
        center_iter = center_only
    else:
        center_iter = itertools.repeat(center_only)

    for func, center in zip(stat_funcs, center_iter):
        blob_stat = get_blob_stats(new_blobs, image, accum_func=func)
        new_blobs = np.column_stack((new_blobs, blob_stat))
    return new_blobs


def _sort_by_col(arr, col, reverse=False):
    sorted_arr = arr[arr[:, col].argsort()]
    return sorted_arr[::-1] if reverse else sorted_arr


def sort_blobs_by_val(blobs, image, radius_pct=1.0, positive=True):
    """Sort the blobs by their absolute value in the image

    Note: blobs must be in (row, col, sigma) form, not (lat, lon, sigma_ll)

    Inputs:
        blobs: [[row, col, radius, amp],...]
        image: 2D ndarray where blobs were detected
        radius_pct (float): (default 1.) Only use `radius_pct` of the middle of the blob

    Returns:
        tuple[tuple[ndarrays], tuple[floats]]: The pair of (blobs, mags)
    """
    if positive:
        reverse = True
        # func = np.max
        pct_func = lambda x: np.percentile(x, 97)
        func = lambda x: gauss_weighted_func(x, func=pct_func)
    else:
        reverse = False
        # func = np.min
        pct_func = lambda x: np.percentile(x, 3)
        func = lambda x: gauss_weighted_func(x, func=pct_func)

    # blob_vals = get_blob_stats(blobs, image, radius_pct=radius_pct, accum_func=func)
    blob_vals = get_blob_stats_weighted(blobs, image, accum_func=func)

    blobs_with_mags = np.column_stack((blobs, blob_vals))
    # Sort rows based on the last column, blob_mag, and in reverse order
    return _sort_by_col(blobs_with_mags, -1, reverse=reverse)


def eig_ratio(patch, blob=None):
    from skimage.feature.corner import _symmetric_image, hessian_matrix

    sigma = sigma_from_blob(blob=blob, patch=patch)

    hessian_mat_array = _symmetric_image(hessian_matrix(patch, sigma))
    det = np.linalg.det(hessian_mat_array)
    trace = np.trace(hessian_mat_array, axis1=2, axis2=3)
    return trace ** 2 / det


def find_sigma_idxs(blobs, sigma_list):
    """Finds which sigma each blob uses by its index in sigma_list

    Assumes blobs already like (r, c, radius,...), where radius=sqrt(2) * sigma"""
    idxs = np.searchsorted(sigma_list, blobs[:, 2] / np.sqrt(2), "left")
    # Clip in case we are passed something larger than any sigma_list
    return np.clip(idxs, 0, len(sigma_list) - 1)


def blobs_to_latlon(blobs, rsc_data=None, filename=None, radius_resolution=1):
    """Converts (y, x, sigma, ...) format to (lat, lon, sigma_latlon, ...)

    Uses the dem x_step/y_step data, or data from GDAL-readable `filename`,
    to rescale blobs so that appear on an image using lat/lon
    as the `extent` argument of imshow.
    """
    if filename is not None:
        return _blobs_to_latlon_rio(blobs, filename, radius_resolution)
    elif rsc_data is not None:
        return _blobs_to_latlon_rsc(blobs, rsc_data, radius_resolution)
    else:
        raise ValueError("Must provide either filename or rsc_data")


def save_blobs_as_geojson(
    outfile, blobs=None, rsc_data=None, rio_filename=None, blobs_ll=None, **kwargs
):
    if blobs_ll is None:
        blobs_ll = blobs_to_latlon(blobs, rsc_data=rsc_data, filename=rio_filename)

    with open(outfile, "w") as f:
        f.write(geojson.dumps(blob_to_geojson(blobs_ll, **kwargs)))


def _blobs_to_latlon_rio(blobs, filename, radius_resolution):
    import rioxarray
    import rasterio as rio

    with rioxarray.open_rasterio(filename) as ds:
        trans = ds.rio.transform()
        x_size = trans[0]
        # y_size = -trans[4]
        lon_lat_arr = np.array(rio.transform.xy(trans, blobs[:, 0], blobs[:, 1])).T

    # print(f"{x_size = }, {y_size = }")
    # If the blob radius is km, need to divide by that, then mult by the degrees factor
    r_factor = x_size / radius_resolution
    print(lon_lat_arr.shape)

    out_blobs = blobs.copy()
    # Make the lat/lon the first two columns
    out_blobs[:, 0] = lon_lat_arr[:, 1]
    out_blobs[:, 1] = lon_lat_arr[:, 0]
    out_blobs[:, 2] *= r_factor
    return out_blobs


def _blobs_to_latlon_rsc(blobs, rsc_data, radius_resolution):
    blobs_latlon = []
    for blob in blobs:
        row, col, r = blob[:3]
        if rsc_data:
            lat, lon = apertools.latlon.rowcol_to_latlon(row, col, rsc_data)
            new_radius = r * rsc_data["x_step"] / radius_resolution
            blobs_latlon.append((lat, lon, new_radius) + tuple(blob[3:]))

    return np.array(blobs_latlon)


def blobs_to_rowcol(blobs, blob_info):
    """Converts (lat, lon, sigma, val) format to (row, col, sigma_latlon, val)

    Inverse of blobs_to_latlon function
    """
    blobs_rowcol = []
    for blob in blobs:
        lat, lon, r, val = blob
        lat, lon = apertools.latlon.latlon_to_rowcol(lat, lon, blob_info)
        old_radius = r / blob_info["x_step"]
        blobs_rowcol.append((lat, lon, old_radius, val))

    return np.array(blobs_rowcol)


def blob_to_geojson(blobs_ll, circle_points=20, extra_columns=["amplitude"]):
    blob_polygons = []
    # TODO write in amp attr
    for lat, lon, rad_deg in blobs_ll[:, :3]:
        p = shapely.geometry.Point([lon, lat])
        d = apertools.latlon.latlon_to_dist([lat, lon], [lat, lon + rad_deg])  # m
        angles = np.linspace(0, 360, circle_points)
        polygon = geog.propagate(p, angles, d)
        blob_polygons.append(
            shapely.geometry.mapping(shapely.geometry.Polygon(polygon))
        )

    gj_list = [geojson.Feature(geometry=gj) for gj in blob_polygons]
    for blob, feat in zip(blobs_ll, gj_list):
        for idx, col in enumerate(extra_columns):
            feat["properties"][col] = blob[3 + idx]
    return geojson.FeatureCollection(gj_list)


def img_as_uint8(img, vmin=None, vmax=None):
    # Handle invalids with masked array, set it to 0
    out = np.ma.masked_invalid(img).filled(0)
    if vmin is not None or vmax is not None:
        out = np.clip(out, vmin, vmax)
    return skimage.img_as_ubyte(skimage.exposure.rescale_intensity(out))


def cv_bbox_to_extent(bbox):
    """convert opencv (top x, top y, w, h) to (left, right, bot, top)

    For use in latlon.intersection_over_union"""
    x, y, w, h = bbox
    # Note: these are row, col pixels, but we still do y + h so that
    # top is a larger number
    return (x, x + w, y, y + h)


def bbox_to_coords(bbox, cv_format=False):
    if cv_format:
        bbox = cv_bbox_to_extent(bbox)
    left, right, bot, top = bbox
    return ((left, bot), (right, bot), (right, top), (left, top), (left, bot))


def regions_to_shapes(regions):
    return [shapely.geometry.Polygon(r) for r in regions]


def _box_is_bad(bbox, min_pix=3, max_ratio=5):
    """Returns true if (x, y, w, h) box is too small or w/h is too oblong"""
    x, y, w, h = bbox
    return w < min_pix or h < min_pix or w / h > max_ratio or h / w > max_ratio


# # TODO: GET THE KM TO PIXEL CONVERSION
# # TODO: maybe make a gauss pyramid, then only do small MSERs
# # TODO: does this work on negative regions at same time as pos? try big path85
# def find_mser_regions(img, min_area=50):
#     import cv2 as cv
#     mser = cv.MSER_create()
#     # TODO: get minarea, maxarea from some km set, convert to pixel
#     mser.setMinArea(220)
#     # mser.setMaxArea(220)
#     regions, bboxes = mser.detectRegions(img)
#     regions, bboxes = prune_regions(regions, bboxes, overlap=0.5)
#     return regions, bboxes


#
# def combine_hulls(points1, points2):
# h = ConvexHull(np.vstack((points1, points2)))
# Or maybe
# h.add_points(points2)
# h.close()
# h = ConvexHull(points1)
# h.add_points(points2)
# return h


def prune_regions(regions, bboxes, overlap_thresh=0.5):
    """Takes in mser regions and bboxs, merges nested regions into the largest"""
    # tup = (box, region)
    # bb = [cv_bbox_to_extent(b) for b in bboxes]
    sorted_bbox_regions = sorted(
        zip(bboxes, regions),
        key=lambda tup: apertools.latlon.box_area(cv_bbox_to_extent(tup[0])),
        reverse=True,
    )
    # Break apart again
    sorted_bboxes, sorted_regions = zip(*sorted_bbox_regions)

    # TODO: i'm just eliminating now (like nonmax suppression)... do i
    # want to merge by combining hulls?
    eliminated_idxs = set()
    # Start with current largest box, check all smaller for overlaps to eliminate
    for idx, big_box in enumerate(sorted_bboxes):
        if idx in eliminated_idxs:
            continue
        bbig = cv_bbox_to_extent(big_box)
        for jdx, sbox in enumerate(sorted_bboxes[idx + 1 :], start=idx + 1):
            if jdx in eliminated_idxs:
                continue
            bsmall = cv_bbox_to_extent(sbox)
            if (
                apertools.latlon.intersect_area(bbig, bsmall)
                / apertools.latlon.box_area(bsmall)
            ) > overlap_thresh:
                eliminated_idxs.add(jdx)

    # Now get the non-eliminated indices
    all_idx = range(len(sorted_bboxes))
    remaining = list(set(all_idx) - set(eliminated_idxs))
    # Converts to np.array to use fancy indexing
    return list(np.array(sorted_regions)[remaining]), np.array(sorted_bboxes)[remaining]


def gaussian_filter_nan(image, sigma, mode="constant", **kwargs):
    """Apply a gaussian filter to an image with NaNs (avoiding all nans)

    The scipy.ndimage `gaussian_filter` will make the output all NaNs if
    any of the pixels in the input that touches the kernel is NaN

    Source:
    https://stackoverflow.com/a/36307291

    Args:
        image: ndarray with nans to filter
        sigma: filter size, passed into gaussian_filter
        **kwargs: passed into gaussian_filter

    Returns:

    """
    if np.sum(np.isnan(image)) == 0:
        return gaussian_filter(image, sigma=sigma, mode=mode, **kwargs)

    V = image.copy()
    nan_idxs = np.isnan(image)
    V[nan_idxs] = 0
    V_filt = gaussian_filter(V, sigma, **kwargs)

    W = np.ones(image.shape)
    W[nan_idxs] = 0
    W_filt = gaussian_filter(W, sigma, **kwargs)

    return V_filt / W_filt


def sigma_from_blob(blob=None, patch=None):
    """Back out what the sigma is based on size of patch or blob radius

    Uses the fact that r = sqrt(2)*sigma
    """
    if blob is not None:
        radius = blob[2]
    elif patch is not None:
        rows, _ = patch.shape
        radius = rows // 2
    else:
        raise ValueError("Need blob or patch for sigma_from_blob")
    return radius / np.sqrt(2)

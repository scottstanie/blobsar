"""utils.py: Functions for finding blobs in deformation maps
"""
import collections
import itertools
import numpy as np
from scipy.ndimage import gaussian_filter


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
        center_only (bool): (default False) Only get the value of the center pixel
            of the blob
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
    """Create a Gaussian kernal with std. dev `sigma` to weight a patch"""
    ny, nx = shape
    if sigma is None:
        sigma = min(nx, ny) / 4
    y = (np.arange(ny) - ny // 2).reshape((-1, 1))
    x = (np.arange(nx) - nx // 2).reshape((1, -1))
    C = 1 / (2 * np.pi * sigma**2) if pdf else 1
    return C * np.exp(-(x**2 + y**2) / (2 * sigma**2))


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


def find_sigma_idxs(blobs, sigma_list):
    """Finds which sigma each blob uses by its index in sigma_list

    Assumes blobs already like (r, c, radius,...), where radius=sqrt(2) * sigma"""
    idxs = np.searchsorted(sigma_list, blobs[:, 2] / np.sqrt(2), "left")
    # Clip in case we are passed something larger than any sigma_list
    return np.clip(idxs, 0, len(sigma_list) - 1)


def blobs_to_latlon(blobs, filename=None, radius_resolution=1):
    """Converts (y, x, sigma, ...) format to (lat, lon, sigma_latlon, ...)

    Uses the dem x_step/y_step data, or data from GDAL-readable `filename`,
    to rescale blobs so that appear on an image using lat/lon
    as the `extent` argument of imshow.
    """
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


def save_blobs_as_geojson(
    outfile, blobs=None, rsc_data=None, rio_filename=None, blobs_ll=None, **kwargs
):
    import geojson

    if blobs_ll is None:
        blobs_ll = blobs_to_latlon(blobs, rsc_data=rsc_data, filename=rio_filename)

    with open(outfile, "w") as f:
        f.write(geojson.dumps(blob_to_geojson(blobs_ll, **kwargs)))


def blob_to_geojson(blobs_ll, circle_points=20, extra_columns=["amplitude"]):
    import geog
    import shapely.geometry
    import geojson

    blob_polygons = []
    # TODO write in amp attr
    for lat, lon, rad_deg in blobs_ll[:, :3]:
        p = shapely.geometry.Point([lon, lat])
        # TODO: Prob a better way to make a circle around the lat/lon coordinates
        d = latlon_to_dist([lat, lon], [lat, lon + rad_deg])  # m
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


def latlon_to_dist(lat_lon_start, lat_lon_end):
    """Find the distance between two lat/lon points on Earth [in meters]

    lats and lons are in degrees, WGS84 ellipsoid is used
    wrapper around pyproj.Geod for older compatibility

    Args:
        lat_lon_start (tuple[int, int]): (lat, lon) in degrees of start
        lat_lon_end (tuple[int, int]): (lat, lon) in degrees of end

    Returns:
        float: distance between two points in meters

    Examples:
        >>> round(latlon_to_dist((38.8, -77.0), (38.9, -77.1)))
        14092
    """
    from pyproj import Geod

    WGS84 = Geod(ellps="WGS84")
    lat1, lon1 = lat_lon_start
    lat2, lon2 = lat_lon_end
    return WGS84.line_length((lon1, lon2), (lat1, lat2))


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

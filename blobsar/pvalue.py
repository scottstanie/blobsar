import os

import numpy as np
import toml
from scipy.stats import gaussian_kde

import blobsar.core as core
from blobsar.constants import AMP_COL, FILT_AMP_COL, SIG_COL
from blobsar.logger import get_log, log_runtime

logger = get_log()


@log_runtime
def find_blob_pvalues(
    image,
    cfg_fname=None,
    sim_dir=None,
    kde_file="kde_ampcol_{amp_col}.npz",
    blob_params=None,
    resolution=None,
    kde_bw_method=0.3,
    display_kde=False,
    num_sar_dates=None,
    stacking_factor=1.0,
    amp_col=AMP_COL,
):
    import pandas as pd

    if cfg_fname is not None:
        cfg = toml.load(cfg_fname)
        sim_blob_params = cfg["blobsar"]
        if blob_params:
            # If we pass extra to update:
            sim_blob_params.update(blob_params)
        blob_params = sim_blob_params
        resolution = cfg["troposim"]["resolution"]

    if blob_params is None:
        raise ValueError("Must provide blob_params or cfg_fname")
    if resolution is None:
        raise ValueError("Must provide resolution or cfg_fname")

    _, Z, extent = find_sim_pdf(
        cfg_fname=cfg_fname,
        sim_dir=sim_dir,
        kde_file=kde_file,
        resolution=resolution,
        kde_bw_method=kde_bw_method,
        display_kde=display_kde,
        amp_col=amp_col,
    )

    logger.info("Finding the blobs within image")
    image_blobs, _ = core.find_blobs(
        image,
        verbose=1,
        **blob_params,
    )
    blobs_km = image_blobs.copy()
    blobs_km[:, SIG_COL] *= resolution / 1000

    if blobs_km.shape[1] == 4:
        df = pd.DataFrame(blobs_km, columns=["row", "col", "r", "amp"])
    else:
        df = pd.DataFrame(blobs_km, columns=["row", "col", "r", "filtamp", "amp"])
    if num_sar_dates is not None:
        stacking_factor = num_sar_dates / cfg["troposim"]["num_days"]
        logger.info(f"Stacking factor for adjusting simulated PDF: {stacking_factor}")

    pvalues = [
        find_pvalue(
            Z,
            extent=extent,
            blob=b,
            stacking_factor=stacking_factor,
            amp_col=amp_col,
        )[2]
        for b in blobs_km
    ]
    df["pvalue"] = pvalues
    return df


def find_sim_pdf(
    cfg_fname=None,
    sim_dir=None,
    kde_file="kde_ampcol_{amp_col}.npz",
    resolution=None,
    kde_bw_method=0.3,
    display_kde=False,
    amp_col=AMP_COL,
):
    from blobsar.simulation import load_all_blobs

    if cfg_fname is not None:
        cfg = toml.load(cfg_fname)
        resolution = cfg["troposim"]["resolution"]

    if resolution is None:
        raise ValueError("Must provide resolution or cfg_fname")

    if sim_dir is None:
        sim_dir = os.path.dirname(cfg_fname)

    sim_blobs = load_all_blobs(sim_dir)
    logger.info("Creating KDE from simulation detected blobs")
    kde_file = os.path.join(sim_dir, kde_file.format(amp_col=amp_col))
    if os.path.exists(kde_file):
        logger.info(f"Loading saved KDE from {kde_file}")
        with np.load(kde_file, allow_pickle=True) as f:
            Z = f["Z"]
            extent = f["extent"]
    else:
        Z, _, extent = kde(
            sim_blobs,
            bw_method=kde_bw_method,
            resolution=resolution,
            display=display_kde,
            amp_col=amp_col,
        )
        logger.info(f"Saving KDE to {kde_file}")
        np.savez(kde_file, Z=Z, extent=extent)
    return sim_blobs, Z, extent


def cdf_kalepy(pts, pdf, test_points):
    import kalepy as kale
    from scipy.interpolate import RegularGridInterpolator

    cdf = kale.utils.cumtrapz(pdf, pts)
    _cdf_grid = (pts, cdf)
    _cdf_func = RegularGridInterpolator(*_cdf_grid, bounds_error=False, fill_value=None)
    return _cdf_func(test_points)


def find_pvalue(
    Z,
    r=None,
    a=None,
    blob=None,
    extent=None,
    r_max=1,
    a_max=1,
    stacking_factor=1,
    p0_factor=1.0,
    amp_col=AMP_COL,
):
    """Find the right-tail probability of a blob.

    Args:
        Z (ndarray[float]): 2D array of Z values from `kde`
        r (float, optional): radius of blob
        a (float, optional): amplitude of blob (cm)
        blob (list[float], optional): blob result, where last 2 entries=[r, a]
            or, 2D array, each row is a blob result
        extent (tuple[float], optional): extent of Z array (r_min, r_max, a_min, a_max)
            Results from `plot.kde`
        r_max (int, optional): largest `r` value for Z, corresponding
            to Z.shape[0]. Alternative to extent.
        a_max (int, optional): largest `a` value for Z, corresponding
            to Z.shape[0]. Alternative to extent.
        stacking_factor (int, optional): Number of independent measurements used to get
            (r, a), beyond what was used for the Z PDF. Used to scale the PDF down
            (stacking_factor if > 1) so that bigger noise is less likely.
            See notes below for more info.
        p0_factor (float, optional): Increase in p0 for the image where (r, a) was detected,
            beyond what was used for Z. Defaults to 1.
            This factor is used to scale the PDF *up* (if `p0_factor > 1`,
            so that bigger noise is *more* likely.

    Returns:
        Avec (ndarray): Amplitude vector, size = Z.shape[1]
        z_conditional (ndarray): the slice of Z conditioned on radius = r.
            It is all the histogram bins with approximately the same radius,
            with values scaled so they sum to 1.
        pvalue (float): the probability of a blob of similar r, with as big or bigger `a`,
            coming from `Z` (right tailed probability).

    Notes:
        `stacking_factor`: If single-image simulations were used to get Z,
        but N images were stacked to get the (r, a) detection, then the PDF needs to
        reflect this decrease in noise. Noise decreases with the square root of the number of
        independent measurements, so if `stacking_factor` is, e.g., 9, then the PDF
        amplitudes get divided by 3.
        If Z was calculated with 25 dates, and you used 100 dates to get the (r, a),
        you should pass 100 / 25 = 4 as `stacking_factor`.
    """
    if blob is not None and blob.ndim > 1:
        return [
            find_pvalue(
                Z,
                blob=b,
                extent=extent,
                r_max=r_max,
                a_max=a_max,
                p0_factor=p0_factor,
                amp_col=amp_col,
            )
            for b in blob
        ]
    nr, na = Z.shape
    if extent is not None:
        _, r_max, _, a_max = extent
    if blob is not None:
        r, a = blob[-2:] if len(blob) == 4 else blob[[SIG_COL, amp_col]]
    a = np.abs(a)
    idx = int(r / r_max * nr)
    # The Expected PtP amplitude decreases with more dates in the stack
    # assuming that with N dates, we use N/2 independent interferograms
    stacking_decrease = 1 / np.sqrt(stacking_factor)
    # stacking_decrease = 1 / np.sqrt(n_sar_dates / 2)

    # Note: p0_factor should be (study area p0) / (simulation p0)
    # p0_factor>1 means you have a noisier area than you ran the sim on
    # stacking_decrease>1 means you stacked more ifgs than you ran the sim on
    Avec = p0_factor * stacking_decrease * a_max * np.linspace(0, 1, na)

    # Dims of Z are (rad, Amp).
    # First grab a narrow slice near `r`, padding due to uncertain exact radius
    # (sum it so it's 1D, collapse the rad dim)
    zslice = np.sum(Z[idx - 1 : idx + 2, :], axis=0)
    # Now find conditional dist. by dividing by the
    # total area (sum the couple Rad, and all Amp cells)
    z_conditional = zslice / np.sum(zslice)
    # With the 1D marginalized PDF, search for cells larger than `a` and sum
    pvalue = np.sum(z_conditional[Avec > a])
    if np.isnan(pvalue):
        breakpoint()
    return Avec, z_conditional, pvalue


def kde(
    blobs,
    bw_method=0.3,
    resolution=None,
    use_abs=True,
    Z=None,
    kernel=None,
    extent=None,
    zero_start_extent=True,
    display=True,
    amp_col=AMP_COL,
    **plotkwargs,
):
    if use_abs:
        bb = np.abs(blobs.copy())
    else:
        bb = blobs.copy()

    if resolution is not None:
        print(f"Converting blob radii to km with {resolution = } meters")
        bb[:, 2] *= resolution / 1000

    if extent is None:
        print("Finding (radius, amplitude) extents of detected blobs")
        extent = blob_extents(bb, 1.2, amp_col=amp_col)
    if zero_start_extent:
        extent[0] = extent[2] = 0.0
    print(f"{extent = }")
    [xmin, xmax, ymin, ymax] = extent
    xmin_norm = xmin / xmax
    ymin_norm = ymin / ymax
    X, Y = np.mgrid[xmin_norm:1:100j, ymin_norm:1:100j]
    # X, Y = np.mgrid[0:1:100j, 0:1:100j]
    # X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # positions = np.vstack([X.ravel(), Y.ravel()])
    # Evaluate the fitted KDE over a space to find PDF values
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Only need the rad, Amp now
    if bb.shape[1] == 4:
        bb = bb[:, -2:]
    elif bb.shape[1] == 5:
        bb = bb[:, [SIG_COL, amp_col]]
    else:
        raise ValueError("bad shape for blobs")
    # Normalize all blob dimensions to max of 1 for KDE fitting
    bbm = np.max(bb, axis=0)
    bb /= bbm

    if Z is None:
        kernel = gaussian_kde(bb.T, bw_method=bw_method)
        Z = np.reshape(kernel.pdf(positions).T, X.shape)
    # Z = np.reshape(kernel.logpdf(positions).T, X.shape)

    if display:
        import blobsar.plot

        blobsar.plot.plot_kde(Z, extent, resolution=resolution, **plotkwargs)
    return Z, kernel, extent


def blob_extents(blobs, pad=1.2, amp_col=AMP_COL):
    """Find the radius, amplitude extremes for plotting extents"""
    if blobs.shape[1] == 4:
        sigmas = blobs[:, SIG_COL]
        amps = blobs[:, 3]
    elif blobs.shape[1] == 5:
        sigmas = blobs[:, SIG_COL]
        amps = blobs[:, amp_col]
    else:
        raise ValueError("bad shape for blobs")
    # Add pad padding onto the maximum limit
    xmin, xmax = np.min(sigmas), pad * np.max(sigmas)
    ymin, ymax = np.min(amps), pad * np.max(amps)
    extent = [xmin, xmax, ymin, ymax]
    return extent

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import toml
import troposim.igrams
from numpy.polynomial import Polynomial
from tqdm.auto import tqdm
from troposim.turbulence import Psd, PsdStack

import blobsar.core as core
from blobsar import pvalue
from blobsar.constants import AMP_COL, SIG_COL
from blobsar.logger import get_log, log_runtime

logger = get_log()


@dataclass(frozen=True)
class BlobsarParams:
    threshold: float = 0.15
    mag_threshold: float = 0.5
    sigma_bins: int = 3
    min_sigma: Optional[float] = 3.5
    max_sigma: Optional[float] = 110.0
    # Alternatives to min_sigma and max_sigma: specify in km
    min_km: Optional[float] = None
    max_km: Optional[float] = None


@dataclass
class Simulator:
    num_sims: int
    # PsdStack object.
    psd_stack: PsdStack
    # Number of days to simulate (if different from the number of days in the PSD stack).
    num_days: Optional[int] = None
    # shape of the simulated image (if different from the shape of the PSD stack).
    shape: Optional[tuple] = None
    # If the PSD stack was made from interferograms instead of estimates of the
    # SAR troposphere, then this should be set to True.
    from_interferograms: Optional[bool] = False

    blobsar_params: BlobsarParams = BlobsarParams()

    out_dir: Optional[Path] = None
    config_file: Optional[Path] = None

    def __post_init__(self):
        self.resolution = self.psd_stack.resolution

        # Allow the user to pass in a dictionary instead of a BlobsarParams object.
        if isinstance(self.blobsar_params, dict):
            self.blobsar_params = BlobsarParams(**self.blobsar_params)
        if self.num_days is None:
            self.num_days = len(self.psd_stack)
        if self.shape is None:
            self.shape = self.psd_stack[0].shape

        if self.blobsar_params.min_km is not None:
            self.blobsar_params.min_sigma = core._dist_to_sigma(
                self.blobsar_params.min_km, self.resolution
            )
            self.blobsar_params.min_km = None

        if self.blobsar_params.max_km is not None:
            self.blobsar_params.max_sigma = core._dist_to_sigma(
                self.blobsar_params.max_km, self.resolution
            )
            self.blobsar_params.max_km = None

        if self.out_dir is None:
            if self.config_file is not None:
                self.out_dir = self.config_file.parent
            else:
                self.out_dir = "./blobsar_data/"
                logger.info(
                    f"No output directory specified. Using {self.out_dir} instead."
                )
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def to_toml(self, toml_file: Optional[Path] = None):
        if toml_file is None:
            return toml.dumps(self.asdict())

        with open(toml_file, "w") as f:
            toml.dump(self.asdict(), f)

    def asdict(self):
        d = asdict(self)
        d["psd_stack"] = self.psd_stack.asdict(include_psd1d=False)
        return d

    @classmethod
    def from_dict(cls, d):
        bp = BlobsarParams(**d.pop("blobsar_params"))
        psd_stack = PsdStack.from_dict(d.pop("psd_stack"))

        return cls(
            psd_stack=psd_stack,
            blobsar_params=bp,
            **d,
        )

    @classmethod
    def from_toml(cls, toml_file):
        with open(toml_file) as f:
            sim = cls.from_dict(toml.load(f))
        sim.config_file = Path(toml_file)

        return sim

    @log_runtime
    def run(
        self,
        start_idx=1,
        divide_cumulative_by=None,
        blob_template_file="blobs_sim_*.npy",
        seed=None,
    ):
        """Run the simulation.

        Parameters
        ----------
        start_idx : int, optional
            Simulation index to start from, by default 1
        divide_cumulative_by : float, optional
            Divide the cumulative stacked deformation by this value, by default None
        blob_template_file : str, optional
            Template for the blob file names, by default "blobs_sim_*.npy"
        seed : int, optional
            Random seed, by default None
        """
        blob_params = self.blobsar_params
        logger.info(f"Saving output to {self.out_dir = }")

        total_blobs = 0

        for nsim in tqdm(range(start_idx, start_idx + self.num_sims + 1)):
            if self.from_interferograms:
                # Directly simulate the interferograms
                assert len(self.psd_stack) == 1
                cumulative_image = self.psd_stack.simulate(
                    num_days=self.num_days, shape=self.shape, seed=seed
                )
            else:
                igm = troposim.igrams.IgramMaker(
                    psd_stack=self.psd_stack,
                    num_days=self.num_days,
                    shape=self.shape,
                    randomize=True,
                    # Convert meters to cm
                    to_cm=True,
                )
                igram_stack = igm.make_igram_stack(seed=seed)
                # Perform stacking on the ifgs to get a cumulative deformation image
                avg_velocity = igram_stack.sum(axis=0) / igm.temporal_baselines.sum()
                time_span = (igm.sar_date_list[-1] - igm.sar_date_list[0]).days
                cumulative_image = avg_velocity * time_span

            logger.debug(f"{np.ptp(cumulative_image) = }")
            if divide_cumulative_by is not None:
                cumulative_image /= divide_cumulative_by
                logger.debug(f"{np.ptp(cumulative_image) = }")

            blobs, _ = core.find_blobs(
                cumulative_image,
                # verbose=1,
                resolution=self.resolution,
                **asdict(blob_params),
            )
            total_blobs += blobs.shape[0]

            if nsim % 10 == 1 and nsim > start_idx:
                logger.info(
                    f"Found {total_blobs} blobs in {nsim - start_idx} simulations"
                )

            outname = blob_template_file.replace("*", str(nsim))
            np.save(self.out_dir / outname, blobs)

    def get_pdf(
        self,
        kde_file=None,
        kde_bw_method=0.3,
        amp_col=AMP_COL,
        display_kde=False,
        vm_pct=99,
        display_savename=None,
        **plot_kwargs,
    ):
        sim_blobs = load_all_blobs(self.out_dir)
        logger.info("Creating KDE from simulation detected blobs")
        if kde_file is None:
            kde_file = self.out_dir / f"kde_amp_col_{amp_col}_sim.npz"

        if kde_file.exists():
            logger.info(f"Loading saved KDE from {kde_file}")
            with np.load(kde_file, allow_pickle=True) as f:
                Z = f["Z"]
                extent = f["extent"]
        else:
            Z, _, extent = pvalue.kde(
                sim_blobs,
                resolution=self.resolution,
                bw_method=kde_bw_method,
                display=False,
                amp_col=amp_col,
                vm_pct=vm_pct,
            )
            logger.info(f"Saving KDE to {kde_file}")
            np.savez(kde_file, Z=Z, extent=extent)

        if display_kde:
            import blobsar.plot

            fig, ax = blobsar.plot.plot_kde(
                Z,
                extent,
                resolution=self.resolution,
                vm_pct=vm_pct,
                ampcol=amp_col,
                **plot_kwargs,
            )
            if display_savename:
                fig.savefig(display_savename)
        return sim_blobs, Z, extent

    @log_runtime
    def find_blob_pvalues(
        self,
        image,
        kde_bw_method=0.3,
        display_kde=False,
        num_sar_dates=None,
        stacking_factor=1.0,
        amp_col=AMP_COL,
        verbose=0,
    ):
        _, Z, extent = self.get_pdf(
            kde_bw_method=kde_bw_method,
            display_kde=display_kde,
            amp_col=amp_col,
        )

        logger.info("Finding the blobs within image")
        image_blobs, _ = core.find_blobs(
            image,
            verbose=verbose,
            resolution=self.resolution,
            **asdict(self.blobsar_params),
        )
        blobs_km = image_blobs.copy()
        blobs_km[:, SIG_COL] *= self.resolution / 1000

        if blobs_km.shape[1] == 4:
            df = pd.DataFrame(blobs_km, columns=["row", "col", "r", "amp"])
        else:
            df = pd.DataFrame(blobs_km, columns=["row", "col", "r", "filtamp", "amp"])

        if num_sar_dates is not None:
            stacking_factor = num_sar_dates / self.num_days
            logger.info(
                f"Stacking factor for adjusting simulated PDF: {stacking_factor}"
            )

        pvalues = [
            pvalue.find_pvalue(
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

    @classmethod
    def _from_toml_old(cls, toml_file):
        """Parsing the original format of the simulations.

        This is deprecated and will be removed in the future.
        """
        with open(toml_file) as f:
            sim_params = toml.load(f)

        blob_params = sim_params.pop("blobsar")
        bp = BlobsarParams(**blob_params)

        tropo_params = sim_params.pop("troposim")
        # convert the arrays to numpy arrays of floats
        tropo_params["p0_arr"] = np.array(tropo_params["p0_arr"], dtype=float)
        tropo_params["beta_arr"] = [
            Polynomial(np.array(b).astype(float)) for b in tropo_params["beta_arr"]
        ]
        ps = PsdStack(
            [
                Psd.from_p0_beta(
                    p0=p0,
                    beta=tropo_params["beta_arr"],
                    resolution=tropo_params["resolution"],
                    shape=tropo_params["shape"],
                    freq0=1e-4,
                )
                for p0 in tropo_params["p0_arr"]
            ]
        )
        tropo_params["psd_stack"] = ps

        return cls(
            blobsar_params=bp,
            config_file=toml_file,
            **tropo_params,
        )


def load_all_blobs(sim_dir, blob_template_file="blobs_sim_*.npy"):
    """Load all blob files resulting from simulation runs"""
    datapath = Path(sim_dir)

    all_blob_files = sorted(datapath.glob(blob_template_file))
    logger.info(f"{len(all_blob_files)} image runs from simulation")
    all_blob_list = [np.load(f) for f in all_blob_files]
    logger.info(f"Shapes: {[bb.shape for bb in all_blob_list[:4]]}...")
    all_blobs = np.vstack(all_blob_list)
    logger.info(f"{len(all_blobs)} total blobs")
    return all_blobs


def load_psd1d(psd_npz_file):
    """Load the result of `troposim.analysis.get_all_1d_psd_curves`"""
    with np.load(psd_npz_file, allow_pickle=True) as f:
        p0_arr = f["p0_arr"]
        beta_arr = f["beta_arr"]
        beta_mean = f.get("beta_mean")
        freq_arr = f["freq_arr"]
        psd1d_arr = f["psd1d_arr"]

    return p0_arr, beta_arr, beta_mean, freq_arr, psd1d_arr


######################################
from rich.console import Console
from rich.table import Table


def mark_true_matches(
    df, blob_locations, distance_threshold: float | None = None, copy: bool = True
):
    """
    Add a 'is_correct' column to the DataFrame indicating whether each detected blob
    corresponds to an actual blob location.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with detected blobs, containing at least 'row' and 'col' columns.
    blob_locations : list of tuple
        List of (row, col, sigma) tuples representing the true blob locations.
    distance_threshold : float, optional
        Maximum Euclidean distance between detected and actual blob centers
        to consider it a match.
        If None, uses half the closest blob sigma.
    copy : bool
        Operate on and return a copy of `df`.
        Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional 'correct' and 'true_location' columns.
    """
    import numpy as np

    # Make a copy to avoid modifying the original
    result_df = df.copy() if copy else df

    # Create new columns
    result_df["is_correct"] = False

    # For each detected blob
    for idx, row in result_df.iterrows():
        detected_row, detected_col = row["row"], row["col"]

        min_distance = np.inf

        # Find the closest true blob
        for blob_row, blob_col, blob_sigma in blob_locations:
            distance = np.sqrt(
                (detected_row - blob_row) ** 2 + (detected_col - blob_col) ** 2
            )
            if distance < min_distance:
                min_distance = distance

        # If the closest blob is within the threshold, mark it as correct
        threshold = (
            distance_threshold if distance_threshold is not None else blob_sigma / 2
        )
        result_df.at[idx, "is_correct"] = min_distance <= threshold

    return result_df


def simulate_detections(
    sim,
    psd_stack,
    num_days,
    shape,
    sigma_range=(8, 25),
    simulated_amps=np.arange(0.5, 6, 0.5),
    num_repeats=5,
):
    from troposim.deformation import synthetic
    from scipy import stats

    truths = []
    dfs = []

    for amp in simulated_amps:
        for _ in range(num_repeats):
            current_deformation, blob_locations = synthetic.multiple_gaussians(
                amp_range=(amp, amp),
                shape=shape,
                num_blobs=20,
                sigma_range=sigma_range,
                avoid_overlap=True,
                min_distance_factor=1,
            )
            truths.append(blob_locations)

            igm = troposim.igrams.IgramMaker(
                psd_stack=psd_stack,
                num_days=num_days,
                shape=shape,
                to_cm=True,
            )
            igram_stack = igm.make_igram_stack()

            # Perform stacking on the noise images, then add to the deformation
            time_span = (igm.sar_date_list[-1] - igm.sar_date_list[0]).days

            avg_velocity_noise = igram_stack.sum(axis=0) / igm.temporal_baselines.sum()
            noise_image = avg_velocity_noise * time_span
            print(
                f"{np.ptp(noise_image)/2 = :.2f}, deformation {amp = }, SNR = {amp / np.max(noise_image):.2f}"
            )
            cumulative_image = noise_image + current_deformation

            df = sim.find_blob_pvalues(cumulative_image)
            df_filtamp = sim.find_blob_pvalues(cumulative_image, amp_col=3)

            df_merged = pd.merge(
                df,
                df_filtamp,
                on=("row", "col", "r", "filtamp", "amp"),
                suffixes=("_amp", "_filtamp"),
            )
            df_clean = df_merged[
                (df_merged.pvalue_amp < 0.2) & (df_merged.pvalue_filtamp < 0.2)
            ]
            df_clean.loc[:, "pvalue_harmonic_mean"] = stats.hmean(
                df_clean[["pvalue_amp", "pvalue_filtamp"]], axis=1, weights=[0.8, 0.2]
            )

            dfs.append(mark_true_matches(df_clean, blob_locations))
    return dfs, truths

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


@dataclass
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
        if self.blobsar_params.max_km is not None:
            self.blobsar_params.max_sigma = core._dist_to_sigma(
                self.blobsar_params.max_km, self.resolution
            )

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
            igm = troposim.igrams.IgramMaker(
                psd_stack=self.psd_stack,
                num_days=self.num_days,
                shape=self.shape,
                randomize=True,
                # Convert meters to cm
                to_cm=True,
            )

            igram_stack = igm.make_igram_stack(
                seed=seed,
                # p0_arr=tp.p0_arr,
                # beta=tp.beta,
                # beta_arr=tp.beta_arr,
            )

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
                **asdict(blob_params),
            )
            total_blobs += blobs.shape[0]

            if nsim % 10 == 0:
                logger.info(
                    f"Found {total_blobs} blobs in {nsim - start_idx} simulations"
                )

            outname = blob_template_file.replace("*", str(nsim))
            # outname = f"blobs_sim_{nsim}.npy"
            np.save(self.out_dir / outname, blobs)

        # all_blobs = load_all_blobs(out_dir)
        # combined_path = out_dir / combined_blob_file
        # logger.info(f"Saving combined blob file to {combined_path}")
        # np.save(combined_path, all_blobs)

    def load_all_blobs(self):
        return load_all_blobs(self.out_dir)

    def get_pdf(
        self,
        kde_file=None,
        kde_bw_method=0.3,
        amp_col=AMP_COL,
        display_kde=False,
        vm_pct=99,
        **plot_kwargs,
    ):
        sim_blobs = self.load_all_blobs()
        logger.info("Creating KDE from simulation detected blobs")
        if kde_file is None:
            if not hasattr(self, "kde_file"):
                self.kde_file = self.out_dir / "kde_sim.npz"

        if self.kde_file.exists():
            logger.info(f"Loading saved KDE from {self.kde_file}")
            with np.load(self.kde_file, allow_pickle=True) as f:
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
            logger.info(f"Saving KDE to {self.kde_file}")
            np.savez(self.kde_file, Z=Z, extent=extent)

        if display_kde:
            import blobsar.plot

            blobsar.plot.plot_kde(
                Z, extent, resolution=self.resolution, vm_pct=vm_pct, **plot_kwargs
            )
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
    ):
        _, Z, extent = self.get_pdf(
            kde_bw_method=kde_bw_method,
            display_kde=display_kde,
            amp_col=amp_col,
        )

        logger.info("Finding the blobs within image")
        image_blobs, _ = core.find_blobs(
            image,
            verbose=1,
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

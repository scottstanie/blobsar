import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import toml
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

import troposim.igrams
from troposim.turbulence import Psd, PsdStack
import blobsar.core as core
from blobsar.logger import get_log, log_runtime

logger = get_log()


@dataclass
class BlobsarParams:
    threshold: float = 0.15
    mag_threshold: float = 0.5
    sigma_bins: int = 3
    min_sigma: float = 3.5
    max_sigma: float = 110.0


@dataclass
class TroposimParams:
    # PsdStack object.
    psd_stack: PsdStack
    # Number of days to simulate (if different from the number of days in the PSD stack).
    num_days: Optional[int] = None
    # shape of the simulated image (if different from the shape of the PSD stack).
    shape: Optional[tuple] = None

    def asdict(self):
        d = asdict(self)
        d["psd_stack"] = self.psd_stack.asdict(include_psd1d=False)
        return d


@dataclass
class Simulator:
    num_sims: int
    blobsar_params: BlobsarParams
    troposim_params: TroposimParams
    config_file: Optional[Path] = None

    def to_toml(self, toml_file: Optional[Path] = None):
        if toml_file is None:
            return toml.dumps(self.asdict())

        with open(toml_file, "w") as f:
            toml.dump(self.asdict(), f)

    def asdict(self):
        d = asdict(self)
        d["troposim_params"] = self.troposim_params.asdict()
        return d

    @classmethod
    def from_dict(cls, d):
        num_sims = d["num_sims"]
        bp = BlobsarParams(**d["blobsar_params"])
        tp = TroposimParams(**d["troposim_params"])

        psd_stack = PsdStack.from_dict(tp.psd_stack)
        tp.psd_stack = psd_stack

        return cls(num_sims, bp, tp)

    @classmethod
    def from_toml(cls, toml_file):
        with open(toml_file) as f:
            sim_params = toml.load(f)
        num_sims = sim_params.get("num_sims")

        bp = BlobsarParams(**sim_params.pop("blobsar_params"))

        tp = TroposimParams(**sim_params["troposim_params"])
        psd_stack = PsdStack.from_dict(tp.psd_stack)
        tp.psd_stack = psd_stack

        return cls(
            num_sims=num_sims,
            blobsar_params=bp,
            troposim_params=tp,
            config_file=toml_file,
        )

    @classmethod
    def from_toml_old(cls, toml_file):
        with open(toml_file) as f:
            sim_params = toml.load(f)

        num_sims = sim_params.get("num_sims")
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
        tp = TroposimParams(
            psd_stack=ps,
            num_days=tropo_params.get("num_days"),
        )

        return cls(
            num_sims=num_sims,
            blobsar_params=bp,
            troposim_params=tp,
            config_file=toml_file,
        )

    @log_runtime
    def run(
        self,
        out_dir=None,
        start_idx=1,
        divide_cumulative_by=None,
        combined_blob_file="blobs_combined.npy",
        blob_template_file="blobs_sim_*.npy",
        seed=None,
    ):
        """Run the simulation.

        Parameters
        ----------
        out_dir : str, optional
            Output directory, by default None
        start_idx : int, optional
            Simulation index to start from, by default 1
        divide_cumulative_by : float, optional
            Divide the cumulative stacked deformation by this value, by default None
        combined_blob_file : str, optional
            Name of the combined blob file, by default "blobs_combined.npy"
        blob_template_file : str, optional
            Template for the blob file names, by default "blobs_sim_*.npy"
        seed : int, optional
            Random seed, by default None
        """
        tp = self.troposim_params
        blob_params = self.blobsar_params
        if out_dir is None:
            if self.config_file is not None:
                out_dir = self.config_file.parent
            else:
                out_dir = "./data/"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving output to {out_dir = }")

        for nsim in tqdm(range(start_idx, start_idx + self.num_sims + 1)):
            logger.info(
                f"running simulation {nsim - start_idx + 1} out of {self.num_sims}"
            )

            igm = troposim.igrams.IgramMaker(
                psd_stack=tp.psd_stack,
                num_days=tp.num_days,
                shape=tp.shape,
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
            logger.info(f"{np.ptp(cumulative_image) = }")
            if divide_cumulative_by is not None:
                cumulative_image /= divide_cumulative_by
                logger.info(f"{np.ptp(cumulative_image) = }")

            blobs, _ = core.find_blobs(
                cumulative_image,
                # verbose=1,
                **asdict(blob_params),
            )
            logger.info(f"Found {blobs.shape[0]} blobs")
            outname = blob_template_file.replace("*", str(nsim))
            # outname = f"blobs_sim_{nsim}.npy"
            np.save(out_dir / outname, blobs)

        all_blobs = load_all_blobs(out_dir)
        combined_path = out_dir / combined_blob_file
        logger.info(f"Saving combined blob file to {combined_path}")
        np.save(combined_path, all_blobs)


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

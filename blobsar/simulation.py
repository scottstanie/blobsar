import os
from pathlib import Path
import toml
import numpy as np

import troposim.igram_sim

import blobsar.core as core
from blobsar.logger import get_log, log_runtime

logger = get_log()


@log_runtime
def run_sim(
    config_file=None,
    default_num_sims=50,
    out_dir=None,
    num_days=30,
    shape_default=(800, 800),
    p0_default=2e-4,
    start_idx=1,
    divide_cumulative_by=None,
    combined_blob_file="blobs_combined.npy",
    blob_template_file="blobs_sim_*.npy",
    seed=None,
):

    if config_file is not None:
        with open(config_file) as f:
            sim_params = toml.load(f)

        num_sims = sim_params.get("num_sims", default_num_sims)

        blob_params = sim_params.pop("blobsar")
        tropo_params = sim_params.pop("troposim")

        # extract turbulence parameters
        resolution = tropo_params.get("resolution")
        p0_default = tropo_params.get("p0_default", p0_default)
        shape = tropo_params.get("shape", shape_default)
        num_days = tropo_params.get("num_days", num_days)
        density = tropo_params.get("density")
        p0_arr = tropo_params.get("p0_arr", [])
        p0_arr = np.array(p0_arr).astype(float)
        divide_cumulative_by = tropo_params.get(
            "divide_cumulative_by", divide_cumulative_by
        )
        beta = np.array(tropo_params.get("beta", [])).astype(float)
        beta_arr = tropo_params.get("beta_arr", [])
        beta_arr = [
            np.polynomial.Polynomial(np.array(b).astype(float)) for b in beta_arr
        ]

    if out_dir is None:
        if config_file is not None:
            out_dir = os.path.dirname(config_file)
        else:
            out_dir = "./data/"
    logger.info(f"Saving output to {out_dir = }")

    for nsim in range(start_idx, start_idx + num_sims + 1):
        logger.info(f"running sim {nsim - start_idx + 1} out of {num_sims}")

        igm = troposim.igram_sim.IgramMaker(
            num_days=num_days,
            resolution=resolution,
            shape=shape,
            p0_default=p0_default,
            density=density,
        )

        igram_stack = igm.make_igram_stack(
            seed=seed,
            p0_arr=p0_arr,
            beta=beta,
            beta_arr=beta_arr,
        )
        # Convert meters to cm
        igram_stack *= 100

        # Perform stacking on the ifgs to get a cumulative deformation image
        avg_velocity = igram_stack.sum(axis=0) / igm.temporal_baselines.sum()
        time_span = (igm.sar_date_list[-1] - igm.sar_date_list[0]).days
        cumulative_image = avg_velocity * time_span
        print(f"{np.ptp(cumulative_image) = }")
        if divide_cumulative_by is not None:
            cumulative_image /= divide_cumulative_by
            print(f"{np.ptp(cumulative_image) = }")

        blobs, _ = core.find_blobs(
            cumulative_image,
            verbose=1,
            **blob_params,
        )
        logger.info(f"Found {blobs.shape[0]} blobs")
        outname = blob_template_file.replace("*", str(nsim))
        # outname = f"blobs_sim_{nsim}.npy"
        np.save(os.path.join(out_dir, outname), blobs)

    all_blobs = load_all_blobs(out_dir)
    combined_path = os.path.join(out_dir, combined_blob_file)
    logger.info(f"Saving combined blob file to {combined_path}")
    np.save(combined_path, all_blobs)


def load_all_blobs(sim_dir, blob_template_file="blobs_sim_*.npy"):
    """Load all blob files resulting from simulation runs"""
    datapath = Path(sim_dir)

    all_blob_files = sorted(datapath.glob(blob_template_file))
    print(f"{len(all_blob_files)} image runs from simulation")
    all_blob_list = [np.load(f) for f in all_blob_files]
    print("Shapes: ", [bb.shape for bb in all_blob_list[:4]], "...")
    all_blobs = np.vstack(all_blob_list)
    print(f"{len(all_blobs)} total blobs")
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

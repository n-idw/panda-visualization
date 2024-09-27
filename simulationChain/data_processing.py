# python IO for ROOT files
import uproot

from tqdm.contrib.concurrent import process_map

# numpy
import numpy as np

import awkward as ak
import yaml
import math
import os


from functools import partial

from utils.data_processing import (
    get_all_mother_ids,
    get_process_ids,
    is_signal_particle,
    get_process_name,
    get_particle_tex_name,
)

import logging
import argparse


def process_row(
    row: ak.Array, signal_mc_ids: list[list[int]], signal_process_codes: list[list[int]]
) -> ak.Array:

    # Dictionary containing the processed data of the row.
    output_dict = {}

    # Get the track ids of particles that leave a signal in the STT.
    unique_track_ids = np.unique(ak.to_numpy(row["STTPoint.fTrackID"]))

    # Write only the particle data for particles that leave a signal in the STT into the output dictionary.
    keys = ["MCTrack.fPx", "MCTrack.fPy", "MCTrack.fPz", "MCTrack.fPdgCode"]
    for key in keys:
        output_dict[key] = ak.Array(row[key])[unique_track_ids]

    # Get the mother ids of all particles.
    mother_ids = get_all_mother_ids(
        mother_ids=row["MCTrack.fMotherID"],
        second_mother_ids=row["MCTrack.fSecondMotherID"],
    )

    # Initialize arrays / lists for different entries in the output dictionary.
    output_dict["is_signal"] = np.empty(len(unique_track_ids), dtype=bool)
    output_dict["particle_name"] = [None] * len(unique_track_ids)
    output_dict["production_process"] = [None] * len(unique_track_ids)

    # Iterate over all unique track ids and get the particle wise information.
    particle_num = 0
    for particle_id in unique_track_ids:
        # Get the PDG MC IDs and VMC process codes of the particle leaving the track
        # and all its mother particles.
        mc_ids, process_codes = get_process_ids(
            process_ids=row["MCTrack.fProcess"],
            mother_ids=mother_ids,
            pdg_ids=row["MCTrack.fPdgCode"],
            particle_id=particle_id,
        )

        # Check if the particle is a signal particle.
        output_dict["is_signal"][particle_num] = is_signal_particle(
            process_mc_ids=mc_ids,
            process_ids=process_codes,
            signal_mc_ids=signal_mc_ids,
            signal_process_ids=signal_process_codes,
        )

        # Get the production process name of the particle.
        output_dict["production_process"][particle_num] = get_process_name(
            process_id=process_codes[-1]
        )

        # Get the particle name in LaTeX format.
        output_dict["particle_name"][particle_num] = (
            "$"
            + get_particle_tex_name(
                pdg_id=np.array(row["MCTrack.fPdgCode"][particle_id], dtype=int)
            )
            + "$"
        )

        # Calculate the transverse momentum.
        output_dict["pt"] = np.sqrt(
            output_dict["MCTrack.fPx"] ** 2 + output_dict["MCTrack.fPy"] ** 2
        )
        # Calculate the absolute momentum.
        output_dict["P"] = np.sqrt(
            output_dict["MCTrack.fPx"] ** 2
            + output_dict["MCTrack.fPy"] ** 2
            + output_dict["MCTrack.fPz"] ** 2
        )
        # Calculate the polar angle theta.
        output_dict["theta"] = np.arctan2(output_dict["pt"], output_dict["MCTrack.fPz"])
        # Calculate the azimuthal angle phi.
        output_dict["phi"] = np.arctan2(
            output_dict["MCTrack.fPy"], output_dict["MCTrack.fPx"]
        )
        # Calculate the pseudorapidity eta.
        output_dict["eta"] = -np.log(np.tan(output_dict["theta"] / 2))

        particle_num += 1

    # Return the output dictionary as an awkward array.
    return ak.Array(output_dict)


def main() -> None:
    """
    Utility to get particle wise information from the PandaRoot simulation data.
    """

    # Parse the command line arguments.
    parser = argparse.ArgumentParser(
        description="Utility to get particle wise information from the PandaRoot simulation data."
    )
    parser.add_argument("input_file", type=str, help="Path to the input ROOT file.")
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where the output parquet file will be saved.",
    )
    parser.add_argument(
        "signal_signature_file",
        type=str,
        help="Path to the YAML file containing the signal signature.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Increases the verbosity: -v = INFO, -vv = DEBUG.",
        default=0,
    )
    parser.add_argument(
        "-n",
        "--num_cpus",
        type=int,
        help="Number of cpu cores to use. Default is the number of cpus of the system minus 1.",
        default=os.cpu_count() - 1,
    )
    parser.add_argument(
        "-s",
        "--step_size",
        type=int,
        help="Memory allocated per step for reading entries from the ROOT file in MB. Default is 100 MB.",
        default=100,
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        type=int,
        help="Number of entries given to each worker for parallel processing. Default is 10.",
        default=10,
    )
    parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        help="Name of the output parquet file. Default is particle_data.parquet.",
        default="particle_data.parquet",
    )
    parser.add_argument(
        "-e",
        "--num_entries",
        type=int,
        help="Number of entries to process. Default is all entries in the ROOT file.",
        default=None,
    )

    # Parse the command line arguments.
    command_line_args = parser.parse_args()

    # Set the logging level based on the verbosity.
    if command_line_args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO, format=" %(levelname)s in %(filename)s :: %(message)s"
        )
    elif command_line_args.verbose >= 2:
        logging.basicConfig(
            level=logging.DEBUG, format=" %(levelname)s in %(pathname)s :: %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format=" %(levelname)s in %(filename)s :: %(message)s",
        )

    # Log the parsed command line arguments.
    if command_line_args.verbose > 0:
        logging.info("Parsed command line arguments:")
        for arg in vars(command_line_args):
            logging.info(f"{arg}: {getattr(command_line_args, arg)}")

    # Read the signal signature from the YAML file.
    with open(command_line_args.signal_signature_file, "r") as file:
        signal_signature = yaml.safe_load(file)

    # Prepare the process_row function with the signal signature.
    prepared_process_row = partial(
        process_row,
        signal_mc_ids=signal_signature["particle_ids"],
        signal_process_codes=signal_signature["process_codes"],
    )

    # Read the "pndsim" TTree from the input ROOT file using uproot.
    sim_tree = uproot.open(
        command_line_args.input_file + ":pndsim",
    )

    # Get the total number of entries in the simulation tree.
    total_entries = sim_tree.num_entries
    logging.info(f"Total number of entries: {total_entries}")

    # Set the number of entries to process.
    if command_line_args.num_entries is None:
        entries_to_process = total_entries
    else:
        entries_to_process = command_line_args.num_entries

    logging.info(f"Entries to process: {entries_to_process}")

    branch_names = [
        "MCTrack.fPx",
        "MCTrack.fPy",
        "MCTrack.fPz",
        "MCTrack.fMotherID",
        "MCTrack.fSecondMotherID",
        "MCTrack.fProcess",
        "MCTrack.fPdgCode",
        "STTPoint.fTrackID",
    ]

    # Calculate the number of entries for a given memory size and the specified branches.
    entries_per_step = sim_tree.num_entries_for(
        expressions=branch_names,
        memory_size=f"{command_line_args.step_size} MB",
    )
    if entries_per_step > entries_to_process:
        logging.info(f"Entries per step: {entries_to_process}")
    else:
        logging.info(f"Entries per step: {entries_per_step}")

    # Calculate the total number of steps.
    steps = math.ceil(entries_to_process / entries_per_step)
    logging.info(f"Total number of steps: {steps}")

    # Iterate in n steps over batches of entries in the simulation tree.
    step = 1
    for batch in sim_tree.iterate(
        expressions=branch_names,
        step_size=f"{command_line_args.step_size} MB",
        entry_stop=entries_to_process,
    ):
        print(f"Processing step {step} of {steps}")

        if step == 1:  # Make a new awkward array of records.
            # The batch of rows gets processed in parallel by the process_row function.
            particle_data = ak.Array(
                process_map(
                    prepared_process_row,
                    batch,
                    max_workers=command_line_args.num_cpus,
                    chunksize=command_line_args.chunk_size,
                )
            )
        else:  # Attach the new rows of the awkward array to the existing one.
            # The batch of rows gets processed in parallel by the process_row function.
            particle_data = ak.concatenate(
                [
                    particle_data,
                    ak.Array(
                        process_map(
                            prepared_process_row,
                            batch,
                            max_workers=command_line_args.num_cpus,
                            chunksize=command_line_args.chunk_size,
                        )
                    ),
                ]
            )

        step += 1

    # Close the ROOT file.
    sim_tree.close()

    # Convert the awkward array to a pandas dataframe.
    particle_df = ak.to_dataframe(
        dict(zip(ak.fields(particle_data), ak.unzip(particle_data)))
    )

    # Reset the index of the dataframe, as ak automatically creates entries and subentries.
    particle_df.reset_index(inplace=True, drop=True)

    logging.info(particle_df.head())

    particle_df.to_parquet(
        f"{command_line_args.output_path}/{command_line_args.output_name}",
        index=False,
    )


if __name__ == "__main__":
    main()

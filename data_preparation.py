import os
import logging
import re
import glob

from zipfile import ZipFile

import pandas as pd
import numpy as np

import visualisation

# Default data location:
# zipped directory named "gunshot-residue.zip" in same parent directory as the current file.
DATASET_PATH = os.path.join(os.getcwd(), "gunshot-residue.zip")

def setup():
    """
    If necessary, extract data.
    """

    if DATASET_PATH.endswith(".zip"):
        extract(DATASET_PATH)

        # Remove .zip extension, use extracted data from this point onwards
        data_path, _ext = os.path.splitext(data_path)
    else:
        data_path = DATASET_PATH

    return os.path.join(data_path, "data")


def extract(zip_path):
    """
    Extract zipped data.
    """

    with ZipFile(zip_path, 'r') as zipped_database:
        logging.info("Extracting zipped dataset, this might take a while..")
        zipped_database.extractall()
        logging.info("Extraction done! Continuing.")


def get_file_number(path):
    """
    Get the file number of the particle table. This enables to sort the
    tables, as the table numbers are not zero-padded in the database.
    """

    match = re.search(r"particle_(?P<ID>\d+).csv", path)
    return int(match.group("ID"))


def read_particles(particle_paths):
    """
    Read the particle data from the several particle tables into a single dataframe.
    """

    particle_paths = sorted(glob.glob(particle_paths), key=get_file_number)
    particle_dataframes = []

    # Initialize dataframe with first particle file
    particles = pd.read_csv(particle_paths[0])
    columns = particles.keys()
    particle_dataframes.append(particles)

    # Read other particle files
    for particle_path in particle_paths[1:]:
        particles = pd.read_csv(particle_path, names=columns, header=None)
        particle_dataframes.append(particles)

    # Merge dataframes
    return pd.concat(particle_dataframes)


def get_stubs_per_ammo(stubs, ammo_names):
    """
    Create a dictionary containing all stub id's for each ammo type.
    """

    stubs_per_ammo = {}
    for (ammo_hash, friendly_name) in ammo_names.items():
        stubs_per_ammo[friendly_name] = np.array(stubs[stubs.ammo_type == ammo_hash]["id"])

    return stubs_per_ammo


def create_dataset(stubs, particles, ammo_names):
    relevance_classes = np.unique(particles["relevance_class"])
    dataset = pd.DataFrame(columns=["stub_id", "ammo_type"] + list(relevance_classes))

    stubs_per_ammo = get_stubs_per_ammo(stubs, ammo_names)

    for friendly_ammo_name in ammo_names.values():
        for stub_id in stubs_per_ammo[friendly_ammo_name]:
            # DATASET ROW (ONE PER STUB):
            # stub_id | ammo_type (friendly) | Ba | BaAl | BaAls | BaCaSi | ..
            stub_row = {
                "stub_id": stub_id,
                "ammo_type": friendly_ammo_name,
            }

            stub_particles = particles[particles.stub_id == stub_id]

            # If a stub has no particles associated to it, discard the stub
            if len(stub_particles) == 0:
                continue

            stub_relevance_classes = stub_particles["relevance_class"].value_counts()

            # Calculate the fraction of stub particles that had a certain relevance class
            # for each possible relevance class
            for relevance_class in relevance_classes:
                fraction = stub_relevance_classes.get(relevance_class, default=0) / len(stub_particles)
                stub_row[relevance_class] = fraction

            dataset.loc[len(dataset.index)] = stub_row

    return dataset


def get_dataset():
    data_path = setup()

    # Load stub table into dataframe
    stubs = pd.read_csv(os.path.join(data_path, "stub.csv"))
    print(f"Total number of stubs: {len(stubs)}")

    # Remove stubs without known ammo type
    stubs = stubs[stubs.ammo_type.notnull()]
    print(f"Number of stubs with known ammo type: {len(stubs)}")

    # Create mapping from ammunition hashes to easily readible numbers
    ammo_hashes = set(stubs["ammo_type"])
    ammo_names = dict(zip(ammo_hashes, range(len(ammo_hashes))))
    print(f"Number of distinct ammunition types: {len(ammo_hashes)}")

    visualisation.stub_histogram(stubs, ammo_names)

    # Read all particles
    particles = read_particles(os.path.join(data_path, "particle_*.csv"))
    print(f"Total number of particles: {len(particles)}")

    # Remove particles of stubs without known ammo type
    particles = particles[particles.stub_id.isin(list(stubs["id"]))]
    print(f"Number of particles with known ammo type: {len(particles)}")

    dataset = create_dataset(stubs, particles, ammo_names)
    print(f"Stubs without particles removed, {len(dataset)} stubs remaining in final dataset")

    return dataset
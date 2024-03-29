import numpy as np
import matplotlib.pyplot as plt

UVA_COLOR = "#bc0031"
FIG_PATH = "figures/"


def stub_histogram(stubs, ammo_names):
    stubs_per_ammo = np.empty(len(ammo_names))
    stubs_per_ammo_filtered = []

    for (ammo_hash, friendly_name) in ammo_names.items():
        relevant_stubs = stubs[stubs.ammo_type == ammo_hash]
        stubs_per_ammo[friendly_name] = len(relevant_stubs)
        if len(relevant_stubs) > 1 and len(relevant_stubs) < 100:
            stubs_per_ammo_filtered.append(len(relevant_stubs))

    fig = plt.figure(figsize=(5,2))
    labels, counts = np.unique(stubs_per_ammo, return_counts=True)
    plt.bar(labels, counts, align='center', color=UVA_COLOR)
    plt.xlabel("# stubs per ammo type")
    plt.ylabel("# ammo types")
    fig.savefig(FIG_PATH + "stubs-per-ammo.pdf", transparent=True)

    fig = plt.figure(figsize=(5,2))
    labels_filtered, counts_filtered = np.unique(stubs_per_ammo_filtered, return_counts=True)
    plt.bar(labels_filtered, counts_filtered, align='center', color=UVA_COLOR)
    plt.gca().set_xticks(range(np.max(labels_filtered)+1))
    plt.yticks([2*i for i in range(10)])
    plt.xlabel("# stubs per ammo type")
    plt.ylabel("# ammo types")
    fig.savefig(FIG_PATH + "stubs-per-ammo-filtered.pdf")
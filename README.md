# StubPredict - Predictive Modelling of Ammunition Type based on Gunshot Residue Data

This project was written as part of the 'Cybercrime, Digital Traces and
Forensic Data Analysis' course of the MSc Forensic Science programme at the
University of Amsterdam.

## Aim
With this project, we aimed to predict the ammunition type of a gunshot residue
(GSR) stub, based on the particles present in this stub. The project makes use
of a publicly available dataset, with data gathered by the Netherlands Forensic
Institute (NFI), which can be found [here](https://github.com/netherlandsforensicinstitute/gunshot-residue).
The latest version of this dataset (at the time of writing) was used: release v0.1.

We want to thank all involved in creating and publishing this dataset for their
work. More information about their research can be found in the accompanying
paper: [T. Matzen et al. "Objectifying evidence evaluation for gunshot residue comparisons using machine learning on criminal case data." Forensic science international 335 (2022): 111293](https://doi.org/10.1016/j.forsciint.2022.111293).

## Usage
First, be sure to install the requirements of this project (`pip install -r requirements.txt`).

The code requires that the GSR dataset is available locally, which can be
downloaded from the [NFI GitHub repository](https://github.com/netherlandsforensicinstitute/gunshot-residue).
The default is that this zipped dataset (named `gunshot-residue.zip`) is stored
in the same parent directory as this project. The code will then automatically
extract the contents of this directory upon runnning the main script.

If you wish to use a different location, be sure to change the `DATASET_PATH`
variable in `data_preparation.py`. If the contents of the dataset have already
been extracted, feel free to remove the `.zip` extension to speed up the code.

Once this setup is done, you can use this project code by running `analysis.py`.
Figures (used in our accompanying report) are reproduced automatically, and
stored in a `figures` subdirectory of this project.

## Contributing
You can contribute to this project by filing
an [issue](https://github.com/mirrevdwal/stubpredict/issues) or opening a
[pull request](https://github.com/mirrevdwal/stubpredict/pulls).

**Note:** contributions will not be included until May 1st 2024, in order
to keep the repository unchanged for grading of the project.

## Licensing
Licensed under either of [Apache Licence](LICENSE-APACHE), Version 2.0 or
[MIT License](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in StubPredict by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
# Gleason XAI:

Paper: [arxiv](https://doi.org/10.48550/arXiv.2410.15012)
---
## System Requirements
Software dependencies can be found in the `conda-env.yml`.

The Software was created and tested on a Linux system with a GPU.

## Use
Set up:
1. Create conda environment using the conda_env.yml with
   ```
   conda env create -f path/to/conda_env.yml
   conda activate GleasonXAI
   ```
3. Set environment variables:
   - `DATASET_LOCATION`: location of the images and the dataframe containing the labels
   - `EXPERIMENT_LOCATION`: location for the log files during experiments 
4. Extract and add TissueMicroarray.com Data at `DATASET_LOCATION / GleasonXAI / TMA / original`
5. Add label data (`final_filtered_explanations_df.csv`) and hierarchy mapping (`label_remapping.json`) at `DATASET_LOCATION / GleasonXAI`
6. Extract and add directory containing model weigths at `DATASET_LOCATION / GleasonXAI` (with directory structure as is)
7. With [download_data.py](download_data.py), download and add the other datasets
8. (if failed due to missing step 4) create the MicronCalibrated data using [create_downscaled_dataset.py](create_downscaled_dataset.py)

ca. 15min (depending on download speed)

---

Use for single image prediction:
1. adjust paths in [single_prediction.ipynb](single_prediction.ipynb)
2. run the notebook

ca. 2min

---

Use for generating paper visualization:
1. run [test.py](test.py) to create predictions on the test set (at least for models in `GleasonFinal2/label_level1/SoftDiceBalanced-{i}/version_0/`).
2. run [evaluate_paper_results.ipynb](evaluate_paper_results.ipynb) to create the visualizations and figures of the paper.

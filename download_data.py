import os
import requests
import zipfile
import tarfile
import glob
from tqdm import tqdm
from pathlib import Path

from create_downscaled_dataset import save_downscaled_TMAs_microns_based

def main():
    target_path = Path(os.environ["DATASET_LOCATION"])/"GleasonXAI"
    original_TMAs = target_path/"TMA"/"original"
    create_calibrated_dataset = True

    if not original_TMAs.exists() or not glob.glob("PR*", root_dir=original_TMAs):
        print("----")
        print(f"PLEASE DOWNLOAD the TissueMicroarray.com Dataset images first and add them to {original_TMAs} to create the micron calibrated images")
        print("----")
        create_calibrated_dataset = False

    calibrated_TMAs = target_path/"TMA"/"MicronsCalibrated"

    os.makedirs(target_path, exist_ok=True)
    os.makedirs(original_TMAs, exist_ok=True)
    os.makedirs(calibrated_TMAs, exist_ok=True)

    g19_train = "https://m209.syncusercontent.com/zip/00ba920b1d8700367e5a42f336a954de/Train%20Imgs.zip?linkcachekey=2312d2d50&pid=00ba920b1d8700367e5a42f336a954de&jid=b9e3a681"
    g19_test = "https://m209.syncusercontent.com/zip/42dac9829c5e8c825fe58b645874c875/Test.zip?linkcachekey=78655dc80&pid=42dac9829c5e8c825fe58b645874c875&jid=42233f24"
    harvard = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP"

    harvard_data_postfixes = ["3IKI3C", "YWA5TT", "RAFKES", "0SMDAH", "QEDF2L", "0W77ZC", "0R6XWD", "L2E0UK", "UFDMZW", "HUEM2D", "YVNUNM", "BSWH3O"]

    harvard_zips = []
    zips = []

    print("Downloading Arvaniti et al. Dataset")
    for postfix in tqdm(harvard_data_postfixes):
        with requests.get(harvard + f"/{postfix}", stream=True) as response:
            if response.ok:
                with open(target_path / f'{postfix}_harvard_train.tar.gz', 'wb') as g19_write:
                    g19_write.write(response.content)
                    harvard_zips.append(target_path / f'{postfix}_harvard_train.tar.gz')
            else:
                print(f"failed to download harvard dataset with postfix {postfix}")
    
    print("Downloading Gleason 19 Challenge Training data")
    with requests.get(g19_train, stream=True) as response:
            if response.ok:
                with open(target_path / 'temp_g19.zip', 'wb') as g19_write:
                    g19_write.write(response.content)
                    zips.append(target_path / 'temp_g19.zip')
            else:
                print("failed to download Gleason19 challenge train set")

    print("Downloading Gleason 19 Challenge Test data")
    with requests.get(g19_test, stream=True) as response:
            if response.ok:
                with open(target_path / 'test_temp_g19.zip', 'wb') as g19_write:
                    g19_write.write(response.content)
                    zips.append(target_path / 'test_temp_g19.zip')
            else:
                print("failed to download Gleason19 challenge test set")
    
    print(f"- unzipping zip files in {target_path} to {original_TMAs}")
    print("-- Gleason 19 Challenge")
    for zip_data in tqdm(zips):
        with zipfile.ZipFile(zip_data, 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.is_dir():
                    continue
                zip_info.filename = os.path.basename(zip_info.filename)
                zip_ref.extract(zip_info, original_TMAs)
        zip_data.unlink()
    

    print(f"-- Arvaniti Harvard")
    for zip_data in tqdm(harvard_zips):
        with tarfile.open(zip_data, 'r:gz') as tar_ref:
            for zip_info in tar_ref.getmembers():
                if zip_info.isdir():
                    continue
                zip_info.name = os.path.basename(zip_info.name)
                tar_ref.extract(zip_info, original_TMAs)
        zip_data.unlink()
    
    if create_calibrated_dataset:
        TARGET_SPACING = 1.39258  # microns/pixel
        print(f"creating micron calibrated dataset in {calibrated_TMAs}")
        save_downscaled_TMAs_microns_based(original_TMAs, calibrated_TMAs, TARGET_SPACING, ".jpg")

if __name__ == "__main__":
    main()
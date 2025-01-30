import argparse
import glob
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from create_downscaled_dataset import save_downscaled_TMAs_microns_based


def unzip_zip_data(zip_ref, target_path):
    for zip_info in zip_ref.infolist():
        if zip_info.is_dir():
            continue
        zip_info.filename = os.path.basename(zip_info.filename)
        zip_ref.extract(zip_info, target_path)


def unzip_with_dirstructure(zip_ref, target_path):
    for zip_info in zip_ref.infolist():
        if zip_info.is_dir():
            os.makedirs(target_path / zip_info.filename, exist_ok=True)
        zip_ref.extract(zip_info, target_path)


def unzip_tissuearray_data(data_zip: Path, target_path, original_TMAs, expl_file, mapping_file, ta_zip_name, weight_files):
    can_do_calibration = True
    if data_zip.suffix == ".zip":
        print(f"- unzipping GleasonXAI data in {data_zip}")
        with zipfile.ZipFile(data_zip, "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if os.path.basename(zip_info.filename) in {expl_file, mapping_file}:
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, target_path)

                elif os.path.basename(zip_info.filename).startswith(ta_zip_name):
                    print("-- unzipping images")
                    zip_ref.extract(zip_info, target_path)
                    with zipfile.ZipFile(target_path / zip_info.filename, "r") as nested_zip_ref:
                        unzip_zip_data(nested_zip_ref, original_TMAs)
                    os.remove(target_path / zip_info.filename)
                elif os.path.basename(zip_info.filename).startswith(weight_files):
                    print("-- unzipping weights")
                    zip_ref.extract(zip_info, target_path)
                    with zipfile.ZipFile(target_path / zip_info.filename, "r") as nested_zip_ref:
                        unzip_with_dirstructure(nested_zip_ref, target_path)

                    os.remove(target_path / zip_info.filename)

    elif data_zip.is_dir():
        label_path = data_zip / expl_file
        if label_path.exists():
            shutil.copy(label_path, target_path / expl_file)
        else:
            print(f"label mapping file not found at {label_path}")
            can_do_calibration = False

        map_path = data_zip / mapping_file
        if map_path.exists():
            shutil.copy(map_path, target_path / mapping_file)
        else:
            print(f"label mapping file not found at {map_path}")
            can_do_calibration = False

        ta_data = data_zip / ta_zip_name
        if ta_data.exists():
            with zipfile.ZipFile(ta_data, "r") as zip_ref:
                unzip_zip_data(zip_ref, original_TMAs)
        else:
            print(f"TissueArray.com data not found at {ta_data}")
            can_do_calibration = False

        w_data = data_zip / weight_files
        if w_data.exists():
            with zipfile.ZipFile(w_data, "r") as zip_ref:
                for weight_zip_info in zip_ref.infolist():
                    zip_ref.extract(weight_zip_info, target_path)
                # unzip_zip_data(zip_ref, target_path)
        else:
            print(f"Model weights not found at {w_data}")
    return can_do_calibration


def download_and_unzip_data(harvard, g19_train, g19_test, target_path, original_TMAs):
    download_successful = True
    unlink_g19_zips = True

    harvard_data_postfixes = ["3IKI3C", "YWA5TT", "RAFKES", "0SMDAH", "QEDF2L", "0W77ZC", "0R6XWD", "L2E0UK", "UFDMZW", "HUEM2D", "YVNUNM", "BSWH3O"]
    harvard_zips = []
    zips = []

    print("Downloading Arvaniti et al. Dataset")
    for postfix in tqdm(harvard_data_postfixes):
        with requests.get(harvard + f"/{postfix}", stream=True) as response:
            if response.ok:
                with open(target_path / f"{postfix}_harvard_train.tar.gz", "wb") as g19_write:
                    g19_write.write(response.content)
                    harvard_zips.append(target_path / f"{postfix}_harvard_train.tar.gz")
            else:
                print(f"failed to download harvard dataset with postfix {postfix}")
                print(f"full url: {harvard + f'/{postfix}'}")
                download_successful = False

    if urlparse(g19_train).scheme in (
        "http",
        "https",
    ):
        print("Downloading Gleason 19 Challenge Training data")
        with requests.get(g19_train, stream=True) as response:
            if response.ok:
                with open(target_path / "temp_g19.zip", "wb") as g19_write:
                    g19_write.write(response.content)
                    zips.append(target_path / "temp_g19.zip")
                    print("downloaded train")
            else:
                print("failed to download Gleason19 challenge train set")
                print(f"full url: {g19_train}")
                download_successful = False

    elif Path(g19_train).exists() and zipfile.is_zipfile(g19_train):
        zips.append(Path(g19_train))
        unlink_g19_zips = False
    else:
        print(f"Gleason19 Train set does not exist or not a zip file: {g19_train}")
        download_successful = False

    if urlparse(g19_test).scheme in (
        "http",
        "https",
    ):
        print("Downloading Gleason 19 Challenge Test data")
        with requests.get(g19_test, stream=True) as response:
            if response.ok:
                with open(target_path / "test_temp_g19.zip", "wb") as g19_write:
                    g19_write.write(response.content)
                    zips.append(target_path / "test_temp_g19.zip")
                    print("downloaded test")
            else:
                print("failed to download Gleason19 challenge test set")
                print(f"full url: {g19_test}")
                download_successful = False

    elif Path(g19_test).exists() and zipfile.is_zipfile(g19_test):
        zips.append(Path(g19_test))
        unlink_g19_zips = False
    else:
        print(f"Gleason19 Test set does not exist or not a zip file: {g19_test}")
        download_successful = False

    print(f"- unzipping zip files to {original_TMAs}")
    print("-- Gleason 19 Challenge")
    for zip_data in tqdm(zips):
        with zipfile.ZipFile(zip_data, "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.is_dir():
                    continue
                zip_info.filename = os.path.basename(zip_info.filename)
                zip_ref.extract(zip_info, original_TMAs)
        zip_data.unlink()

    print(f"-- Arvaniti Harvard")
    for zip_data in tqdm(harvard_zips):
        with tarfile.open(zip_data, "r:gz") as tar_ref:
            for zip_info in tar_ref.getmembers():
                if zip_info.isdir():
                    continue
                zip_info.name = os.path.basename(zip_info.name)
                tar_ref.extract(zip_info, original_TMAs)
        zip_data.unlink()

    return download_successful


def data_setup(manual: bool, download: bool, calibrate: bool, data_zip: Path, g19_test: str, g19_train: str, harvard: str):
    target_path = Path(os.environ["DATASET_LOCATION"]) / "GleasonXAI"
    original_TMAs = target_path / "TMA" / "original"
    calibrated_TMAs = target_path / "TMA" / "MicronsCalibrated"

    expl_file = "final_filtered_explanations_df.csv"
    mapping_file = "label_remapping.json"
    ta_zip_name = "tissuearray_com_data.zip"
    weight_files = "model_weights.zip"

    os.makedirs(target_path, exist_ok=True)
    os.makedirs(original_TMAs, exist_ok=True)
    os.makedirs(calibrated_TMAs, exist_ok=True)

    calibrated_dataset_possible = True
    if manual:
        if not glob.glob("PR*", root_dir=original_TMAs):
            print("----")
            print(f"set to manual setup, but no TissueArray.com data found at {original_TMAs}")
            print(f"PLEASE DOWNLOAD the TissueMicroarray.com Dataset images first and add them to {original_TMAs} to create the micron calibrated images")
            print("----")
            calibrated_dataset_possible = False
        if not (target_path / "final_filtered_explanations_df.csv").exists():
            print(f"set to manual setup, but label file not found at {target_path / expl_file}")
            calibrated_dataset_possible = False
        if not (target_path / "label_remapping.json").exists():
            print(f"set to manual setup, but label mapping file not found at {target_path / mapping_file}")
            calibrated_dataset_possible = False
        if not (target_path / "GleasonFinal2").exists():
            print(f"set to manual setup, but model weights not found at {target_path / 'GleasonFinal2'}")
    else:
        calibrated_dataset_possible &= unzip_tissuearray_data(data_zip, target_path, original_TMAs, expl_file, mapping_file, ta_zip_name, weight_files)

    if download:
        calibrated_dataset_possible &= download_and_unzip_data(harvard, g19_train, g19_test, target_path, original_TMAs)

    if calibrate and calibrated_dataset_possible:
        TARGET_SPACING = 1.39258  # microns/pixel
        print(f"creating micron calibrated dataset in {calibrated_TMAs}")
        save_downscaled_TMAs_microns_based(original_TMAs, calibrated_TMAs, TARGET_SPACING, ".jpg")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--gleasonxai_data")
    parse.add_argument("--manual_xai_data", action="store_true")
    parse.add_argument("--download", action="store_true")
    parse.add_argument("--calibrate", action="store_true")
    parse.add_argument(
        "--gleason19_test",
        default="https://m208.syncusercontent.com/zip/42dac9829c5e8c825fe58b645874c875/Test.zip?linkcachekey=78655dc80&pid=42dac9829c5e8c825fe58b645874c875&jid=f52a164d",
    )
    parse.add_argument(
        "--gleason19_train",
        default="https://m208.syncusercontent.com/zip/00ba920b1d8700367e5a42f336a954de/Train%20Imgs.zip?linkcachekey=2312d2d50&pid=00ba920b1d8700367e5a42f336a954de&jid=607aa8d6",
    )
    parse.add_argument("--arvaniti", default="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP")

    args = parse.parse_args()

    if not "DATASET_LOCATION" in os.environ:
        print("Please set environment vairable DATASET_LOCATION first.")
        exit(1)

    if args.manual_xai_data:
        assert args.gleasonxai_data is not None, "Please specify location of downloaded dataset"
        assert Path(args.gleasonxai_data).exists(), f"File / directory {args.gleasonxai_data} does not exist"

    data_setup(args.manual_xai_data, args.download, args.calibrate, Path(args.gleasonxai_data), args.gleason19_test, args.gleason19_train, args.arvaniti)

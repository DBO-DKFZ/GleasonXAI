import src.gleason_data as gleason_data


def create_final_publication_dataset(path, split, label_level):
    data = gleason_data.GleasonX(path=path, split=split, scaling="MicronsCalibrated", transforms=None,
                                 label_level=label_level, create_seg_masks=True, explanation_file="final_filtered_explanations_df.csv", data_split=[0.7, 0.15, 0.15], tissue_mask_kwargs={"open": False, "close": False, "flood": False})

    return data

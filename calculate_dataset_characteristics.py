import itertools
import os
import collections

import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from src.gleason_data import GleasonX


matplotlib.use('Agg')

label_mapping = {"poorly formed and fused glands": "4 - poorly formed glands",
                     'variable sized well-formed individual and discrete glands': "3 - individual glands",
                     'cords': "5 - cords",
                     'presence of comedonecrosis': "5 - comedonecrosis",
                     'solid groups of tumor cells': "5 - groups of tumor cells",
                     'Cribriform glands': "4 - cribriform glands",
                     'Glomeruloid glands': "4 - glomeruloid glands",
                     'compressed or angular discrete glands': "3 - compressed glands",
                     'single cells': "5 - single cells"}

base_label_mapping = {"single, individual atypical glands separated from each other": "3.01", 
                        "atypical glands with an irregularly separated, ragged, poorly defined edge": "3.02", 
                        "atypical glands are looser than a nodule and are infiltrative": "3.03", 
                        "either minute or large and cyst-like atrophic atypical glands": "3.04", 
                        "atypical glands lying very closely together (with little stroma between adjacent atypical glands)": "3.05", 
                        "well-formed, relatively uniform atypical glands with evenly distributed lumina": "3.06", 
                        "compressed or angular atypical glands": "3.07", 
                        "atypical glands infiltrate between benign glands": "3.08", 
                        "slit-like lumina": "4.01", 
                        "large atypical glands": "4.02", 
                        "irregular contours, jagged edges of atypical glands": "4.03", 
                        "atypical glands fused or grown together into cords or chains": "4.04", 
                        "irregular distribution of lumina": "4.05", 
                        "atypical glands very close together (with little or no stroma)": "4.06", 
                        "cribriform": "4.07",
                        "cribriform larger than a normal prostate gland; tends to fragmentation": "4.08", 
                        "cribriform confluent sheet of contiguous carcinoma cells with multiple glandular lumina that are easily visible at low power (objective magnification 10x)": "4.09", 
                        "cribriform single or fused glandular structures connected to each other (no intervening stroma or mucin)": "4.10", 
                        "hypernephroid pattern": "4.11",
                        "hypernephroid pattern nests of clear cells resembling renal cell carcinoma": "4.12", 
                        "hypernephroid pattern small, hyperchromatic nuclei": "4.13", 
                        "hypernephroid pattern fusion of acini into more solid sheets with the appearance of back-to-back glands without intervening stroma": "4.14", 
                        "glomeruloid pattern": "4.15",
                        "glomeruloid pattern rare small cribriform variant resembling glomerulus structures of kidney": "4.16", 
                        "glomeruloid pattern contains a tuft of cells that is largely detached from its surrounding duct space except for a single point of attachment": "4.17", 
                        "solid tumor cell clusters with nonpolar nuclei around a lumen": "5.01", 
                        "presence of definite comedonecrosis (central necrosis)": "5.02",
                        "presence of definite comedonecrosis (central necrosis) with intraluminal necrotic cells": "5.03", 
                        "presence of definite comedonecrosis (central necrosis) with karyorrhexis within papillary, cribriform spaces": "5.04", 
                        "single cells": "5.05",
                        "single cells forming cords": "5.06", 
                        "single cells with vacuoles (signet ring cells) but without glandular lumina": "5.07"}


def compare_tma_and_explanation_grade(df: pd.DataFrame, output: Path):

    confusion_matrix = np.zeros(shape=(6, 6))
    tma_grade_dict = {'3': 0, '4': 0, '5': 0}

    for tma_name, tma_series in df.groupby('TMA'):
        grade = tma_name.split('.')[0].split('_')[-1]
        if grade.startswith('grade'):
            grade = grade[-1]
        
        tma_grade_dict[grade] += 1

        for expl_grade in tma_series['explanation_lvl_0'].unique():
            confusion_matrix[int(grade)][int(expl_grade)] += 1
        
    
    cm_df = pd.DataFrame(confusion_matrix)
    cm_df = cm_df.drop(columns=[0, 1, 2])
    cm_df = cm_df.drop(index=[0, 1, 2])
    
    print(tma_grade_dict)
    print(cm_df)

    fig, ax = plt.subplots()

    cm_df.to_csv(output / "image_vs_annotation_grade.csv")

    plt.figure(figsize=(3.5,3))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False, square=True, annot_kws={"size": 14}, linewidths=.5)#, kwargs={'alpha': 0.5})
    # plt.title("Confusion Matrix: TMA grade to Explanation Grade")
    plt.xlabel('Annotation Gleason pattern', fontsize=14)
    plt.ylabel('Image score', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(output / "cm_class_comparison.svg", dpi=1000)

    plt.clf()
    plt.cla()
    plt.close()
    

def get_co_occurence(df: pd.DataFrame, output_path: Path, use_sub_expl: bool):
    co_occurences = []
    grade_column = 'explanation_lvl_2' if use_sub_expl == True else 'explanation_lvl_1' if use_sub_expl == False else 'explanation_lvl_0'
    
    grpd = df.groupby(['TMA', 'annotator'])[grade_column].unique().reset_index()

    i = 0
    for name, grp in grpd.groupby('TMA'):
        co_occurences.append(list(set(np.hstack(grp[grade_column].to_numpy()))))
        i += 1
    print(f"{i} TMAs")
    generate_matrix(co_occurences, output_path)


def generate_matrix(co_occurrences_list: list, output_path: Path):
    if not output_path.exists():
        os.makedirs(output_path)

    ccombinations = [list(itertools.combinations(i,2)) for i in co_occurrences_list]
    a = list(itertools.chain.from_iterable((i, i[::-1]) for c_ in ccombinations for i in c_))

    co_occurence_matrix = pd.pivot_table(pd.DataFrame(a).replace(base_label_mapping), index=0, columns=1, aggfunc='size', fill_value=0)
    # pd.pivot_table(pd.DataFrame(a).replace(label_mapping), index=0, columns=1, aggfunc='size', fill_value=0)

    fig, axs = plt.subplots()
    ax = sns.heatmap(data=co_occurence_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
    ax.tick_params(left=False, bottom=False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.title("Co-occurence of Classes")
    plt.tight_layout()
    plt.savefig(output_path / f"cooccurence.png")
    plt.cla()
    plt.clf()
    plt.close()

def get_class_cooccurrence_between_labelers(df: pd.DataFrame, use_sub_expl: bool, combined_grade_images: bool, output_path: Path):
    np_file_out = output_path
    output_path = output_path / "combined_images" if combined_grade_images else output_path / "grade_images"
    labeling_output = output_path / 'sub_expl' if use_sub_expl == True else output_path / 'explanations' if use_sub_expl == False else output_path / 'grade'
    
    if not labeling_output.exists():
        os.makedirs(labeling_output)

    ca_dict = (agreement_occurrence_per_class(df, use_sub_expl, combined_grade_images))
    total_occurence_dict = dict()

    for clz, tuple_list in ca_dict.items():
        combis, nums = np.unique(tuple_list, return_counts=True, axis=0)
        print("class: " + str(clz))
        print(f"total slides with class: {nums.sum()}")
        total_occurence_dict[str(clz)] = nums.sum()
        print("--")

    total_vals = total_occurence_dict.items()
    total_x, total_y = zip(*total_vals)

    if use_sub_expl == False:
        total_x = [label_mapping.get(lbl) for lbl in total_x]
        new_dict = dict(zip(total_x, total_y))
        total_vals = sorted(new_dict.items())
        total_x, total_y = zip(*total_vals)
    elif use_sub_expl == True:
        for lbl in total_x:
            if base_label_mapping.get(lbl) is None:
                print(lbl)
        total_x = [base_label_mapping.get(lbl) for lbl in total_x]
        new_dict = dict(zip(total_x, total_y))
        total_vals = sorted(new_dict.items())
        total_x, total_y = zip(*total_vals)
    else:
        total_x = [str(lbl) for lbl in total_x]
        new_dict = dict(zip(total_x, total_y))
        total_vals = sorted(new_dict.items())
        total_x, total_y = zip(*total_vals)

    if not (np_file_out / "class_dist").exists():
        os.makedirs(np_file_out / "class_dist", exist_ok=True)
    np.save(np_file_out / "class_dist" / f"{'grade' if use_sub_expl == None else 'sub-expl' if use_sub_expl == True else 'expl'}_label.npy", total_x)
    np.save(np_file_out / "class_dist" / f"{'grade' if use_sub_expl == None else 'sub-expl' if use_sub_expl == True else 'expl'}_value.npy", total_y)
    
    figsize = (2.5, 5) if use_sub_expl == None else (7, 5) if use_sub_expl == True else (4,5)

    fig, ax = plt.subplots(figsize=figsize)
    bar_ct = plt.bar(total_x, total_y, width=0.3)
    ylim = (0, 900) if combined_grade_images else (0, 1100)
    plt.ylim(ylim)
    ax.set_xlim(-1, len(total_x))
    ax.bar_label(bar_ct)

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(labeling_output / "class_occurences.png")
    plt.clf()
    plt.cla()
    plt.close()

    print(" -- three rater -- ")
    plot_three_rater_agreement_occurrence(ca_dict, df, labeling_output, raw=use_sub_expl)

def agreement_occurrence_per_class(df: pd.DataFrame, use_sub_expl: bool, image_grouped: bool) -> collections.defaultdict:
    # for each class describe how many cases were with only one annotator, two or three
    tma_id = 'TMA_identifier' if image_grouped else'TMA'

    class_dict = collections.defaultdict(list)
    grade_column = 'explanation_lvl_2' if use_sub_expl else 'explanation_lvl_1' if use_sub_expl == False else 'explanation_lvl_0'
    
    grpd = df.groupby([tma_id, 'annotator'])[grade_column].unique().reset_index()
    i = 0
    for name, grp in grpd.groupby(tma_id):
        explanations_set = set(np.hstack(grp[grade_column].to_numpy()))
        i += 1
        
        series = []
        list_anns = grp[grade_column].tolist()
        for row in list_anns:
            series.append(pd.Series(row))
        new_df = pd.DataFrame(series, index=grp.index)
        for expl in explanations_set:
            class_dict[expl].append((new_df.isin([expl]).values.sum(), len(grp)))
            # class_dict[expl].append((pd.DataFrame(grp[grade_column].tolist(), index=grp.index).isin([expl]).values.sum(), len(grp)))


    print(f"total TMAs: {i}")
    return class_dict


def plot_three_rater_agreement_occurrence(ca_dict: dict, processed_df: pd.DataFrame,  labeling_output: Path, raw = True) -> None:

    all_filtered_dict = dict()
    for clz, tuple_list in ca_dict.items():
        class_name = base_label_mapping[clz] if raw == True else label_mapping[clz] if raw==False else str(clz)
        combis, nums = np.unique(tuple_list, return_counts=True, axis=0)

        filtered_df = filter_three(combis, nums) # should not change the df
        filtered_df.fillna(0)
        output_path = labeling_output

        if not output_path.exists():
            os.makedirs(output_path)

        all_filtered_dict[clz] = filtered_df
    
    all_filtered_dict = { k: v.set_index('x') for k, v in all_filtered_dict.items()}
    rdf = pd.concat(all_filtered_dict, axis=1)
    rdf.columns = rdf.columns.droplevel(-1) 
    rdf = rdf.fillna(0)
    rdf.columns = [base_label_mapping[col] for col in rdf.columns] if raw == True else [label_mapping[col] for col in rdf.columns] if raw == False else rdf.columns
    
    if raw == True:
        for lbl in base_label_mapping.values():
            if lbl not in rdf.columns:
                rdf[lbl] = pd.Series([0, 0, 0], index=rdf.index)

    rdf.loc[0]= processed_df['TMA'].nunique() - rdf.sum()
    rdf = rdf.sort_values('x')
    rdf = rdf.reindex(sorted(rdf.columns), axis=1)
    rdf.to_csv(output_path / f"{'expl' if raw == False else 'sub-expl' if raw == True else 'grade'}_rater_agreement.csv")
    
    fig_size = (7, 15) if raw == True else (10,9) if raw == False else (6.5, 3)
    fig, ax = plt.subplots(figsize=fig_size)
    ax = sns.heatmap(rdf.T, annot=True, fmt='g', cmap='Blues', square=False, cbar=False, annot_kws={"size": 14 if raw == None else 16 }, linewidths=.5)#cbar_kws={"shrink": 0.2})
    #plt.title('co-occurence of label decisions between three raters\n(per label)')
    plt.xlabel('Number of Annotators', fontsize=18) #22
    plt.ylabel('explanation', fontsize=18) #22
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
    plt.tight_layout()
    plt.savefig(output_path / f"hm_three_labeling_agreement_occurences.svg", dpi=1000)
    plt.cla()
    plt.clf()
    plt.close()


def filter_three(combis, nums):
    agreeing_annotators = []
    filtered_nums = []
    for (annotators, total_annotators), num_occurence in zip(combis, nums):
        if total_annotators == 3:
            agreeing_annotators.append(annotators)
            filtered_nums.append(num_occurence)
    
    data = pd.DataFrame(data={'x': agreeing_annotators, 'z': filtered_nums})
    data = data.fillna(0)
    return data


def file_counter(processed_df: pd.DataFrame, combined_image_df: pd.DataFrame) -> None:
    print(f"num. TMAs: {len(combined_image_df.groupby('TMA'))}")
    print(f"num. TIMAs (single grade): {len(processed_df.groupby('TMA'))}")

    annotator_gradeTMA_count, annotator_TMA_count = get_number_of_annotators_per_TMA(processed_df, combined_image_df)

    print(f"per single grade images: \n {annotator_gradeTMA_count}")
    print(f"----- \n per image \n {annotator_TMA_count}")

    single_annotator_TMA = get_TMA_with_One_annotator(combined_image_df)
    print(f"single annotator images: \n {single_annotator_TMA}")
    print("\n---------\n")

    hv, gC, tA = get_num_images_per_dataset(combined_image_df)
    print(f"harvard: {hv}\nGleasonChallenge: {gC}\ntissueArray: {tA}")
    print("\n-----\n")

    hv = get_number_of_annotators_per_TMA_in_DS(processed_df, 'ZT')
    gC = get_number_of_annotators_per_TMA_in_DS(processed_df, 'slide')
    tA = get_number_of_annotators_per_TMA_in_DS(processed_df, 'PR')
    print("annotators per single grade slide:")
    print(f"harvard: {hv}\nGleasonChallenge: {gC}\ntissueArray: {tA}")
    print("\n-----\n")

    hv = get_number_of_annotators_per_TMA_in_DS(combined_image_df, 'ZT')
    gC = get_number_of_annotators_per_TMA_in_DS(combined_image_df, 'slide')
    tA = get_number_of_annotators_per_TMA_in_DS(combined_image_df, 'PR')
    print("annotators per slide:")
    print(f"harvard: {hv}\nGleasonChallenge: {gC}\ntissueArray: {tA}")
    print("\n-----\n")

    hv = get_number_of_annotators_per_TMA_in_DS_for_group(combined_image_df, 'ZT', '19.3')
    gC = get_number_of_annotators_per_TMA_in_DS_for_group(combined_image_df, 'slide', '19.3')
    tA = get_number_of_annotators_per_TMA_in_DS_for_group(combined_image_df, 'PR', '19.3')
    print("annotators per slide in group 19.3:")
    print(f"harvard: {hv}\nGleasonChallenge: {gC}\ntissueArray: {tA}")
    print("\n-----\n")
    


def get_number_of_annotators_per_TMA(df: pd.DataFrame, grade_removed_df: pd.DataFrame):
    grade_TMAs_number = df.groupby('TMA')['annotator'].nunique().value_counts()
    TMAs_number = grade_removed_df.groupby('TMA')['annotator'].nunique().value_counts()

    return grade_TMAs_number, TMAs_number

def get_number_of_annotators_per_TMA_in_DS(df: pd.DataFrame, ds_prefix: str):
    num_df = df[df['TMA'].str.startswith(ds_prefix)]
    return num_df.groupby('TMA')['annotator'].nunique().value_counts()


def get_number_of_annotators_per_TMA_in_DS_for_group(df: pd.DataFrame, ds_prefix: str, group: str):
    num_df = df[df['TMA'].str.startswith(ds_prefix)]
    num_df = num_df[num_df['group'].astype(str).str.startswith(group)]
    return num_df.groupby('TMA')['annotator'].nunique().value_counts()


def get_TMA_with_One_annotator(df: pd.DataFrame):
    TMAs_number = df.groupby('TMA')['annotator'].nunique()
    return TMAs_number[TMAs_number < 2]


def get_num_images_per_dataset(df: pd.DataFrame):
    havard = "ZT"
    gleasonChallenge = "slide"
    tissueArray = "PR"

    return get_num_images_per_prefix(df, havard), get_num_images_per_prefix(df, gleasonChallenge), get_num_images_per_prefix(df, tissueArray)    

def get_num_images_per_prefix(df: pd.DataFrame, prefix: str):
    return len(df[df['TMA'].str.startswith(prefix)].groupby('TMA'))


def shortener(x: str) -> str:
    x_split = x.split('_')
    if len(x_split) == 2:
        part_one = x_split[0]
        part_two = x_split[1].split('.')[0]
        return (part_one + '_' + part_two)
    else:
        return '_'.join(x.split('_')[:-1])
    

def get_num_single_grade(processed_df: pd.DataFrame):
    print(f"76: {get_num_images_per_prefix(processed_df, 'ZT76')}")
    print(f"111: {get_num_images_per_prefix(processed_df, 'ZT111')}")
    print(f"199: {get_num_images_per_prefix(processed_df, 'ZT199')}")
    print(f"204: {get_num_images_per_prefix(processed_df, 'ZT204')}")
    print(f"zong: {get_num_images_per_prefix(processed_df, 'ZT')}")

    print("")

    print(f"482a: {get_num_images_per_prefix(processed_df, 'PR482')}")
    print(f"633a: {get_num_images_per_prefix(processed_df, 'PR633')}")
    print(f"1001: {get_num_images_per_prefix(processed_df, 'PR1001')}")
    print(f"1921b: {get_num_images_per_prefix(processed_df, 'PR1921b')}")
    print(f"1921c: {get_num_images_per_prefix(processed_df, 'PR1921c')}")
    print(f"ta: {get_num_images_per_prefix(processed_df, 'PR')}")

    print("")

    print(f"gl19: {get_num_images_per_prefix(processed_df, 'slide')}")


def get_num_annotator_and_grade(df: pd.DataFrame, file_prefix: str):
    img_rows = df.loc[df['TMA'].str.startswith(file_prefix)]
    print(f"---- {file_prefix}")
    print(f"num_files: {len(img_rows.groupby('TMA'))}")
    print(f"annotators:\n {img_rows.groupby('TMA')['annotator'].unique()}")
    print("--")
    print(f"grades:\n {img_rows.groupby('TMA')['grade'].unique()}")
    print("--")
    print(f"annotated grades:\n {img_rows.groupby('TMA')['explanation_grade'].unique()}")
    print("----\n")


if __name__=="__main__":
    output = Path("./figures")
    in_path = Path(os.environ["DATASET_LOCATION"]) / "GleasonXAI"

    dataset = GleasonX(in_path, split='all', scaling='MicronsCalibrated', label_level=1, create_seg_masks=True, drawing_order='grade_frame_order', explanation_file='final_filtered_explanations_df.csv', data_split=[0.7, 0.15, 0.15], tissue_mask_kwargs={'open': False, 'close':False, 'flood':False})

    processed_df = dataset.df
    combined_images_df = processed_df.copy()
    combined_images_df['TMA'] = combined_images_df['TMA'].apply(shortener)

    get_num_single_grade(combined_images_df)

    # -- compare grade annotations to image grade
    compare_tma_and_explanation_grade(processed_df, output)

    # -- how many times did the annotator use the same labels on the same TMAs -- #
    use_sub_expl = None # True: sub-explanations, False: explanations, None: Gleason patterns
    use_combined_images = True
    labeling_output = output / f"{'grade' if use_sub_expl == None else 'grouped' if use_sub_expl == False else 'expls'}_labeling_agreement"
    
    used_df = combined_images_df if use_combined_images else processed_df

    get_class_cooccurrence_between_labelers(used_df, use_sub_expl, use_combined_images, labeling_output)

    # -- how many files are available -- #
    print("-- available files --")
    file_counter(processed_df, combined_images_df)

    # -- co-occurence matrix
    print('-- co-occurence matrix --')
    get_co_occurence(combined_images_df, output / f"{'grade' if use_sub_expl == None else 'grouped' if use_sub_expl == False else 'expls'}_cooccurence", use_sub_expl)
   
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


logger = logging.getLogger(__name__)

RAW_DATA_DIR = "/home/share/data/1kGP"


def prepare_data_paths():
    raw_data_dir = Path(RAW_DATA_DIR)
    preprocess_path = raw_data_dir / "preprocessed"

    label_file = preprocess_path / "labels.csv"

    assert preprocess_path.exists(), f"Data path does not exist: {preprocess_path}"
    assert label_file.is_file(), f"File does not exist: {label_file}"

    return preprocess_path, label_file


class PreprocessedGenotypeDataset:
    def __init__(self, path_prefix, label_file):
        super().__init__()
        
        self.target_label_name ='population_code'  # 'superpopulation_code'
        self.group_label_name = 'family_id'
        self.notusing_lables = ['IBS,MSL', # only 1 sample
                                # 'GBR', # accuracy 11%
                                # 'ASW', 'ACB', # accuracy ~ 60%
                                # 'GIH', # acuracy < 80%
                                # 'CHB', 'STU', 'ITU',  # accuracy < 90%
                                ]
        self.X = np.load(f"{path_prefix}_matrix.npy")
        self.variant_info_df = pd.read_csv(f"{path_prefix}_variant.csv", dtype = str)

        self.label_df = pd.read_csv(label_file)
        self.y = self.label_df[self.target_label_name]
        logger.info(f"Read data done. X.shape: {self.X.shape}, y.shape: {self.y.shape}, variant_info.shape: {self.variant_info_df.shape}")

        self.drop_notusing_sample(notusing_list= self.notusing_lables)
        self.y_encoded, self.label_mapping = self.encode_y()
        self.groups = self.label_df[self.group_label_name].to_numpy(copy=True)

        logger.info(f"class distribution of '{self.target_label_name}': {self.y.value_counts().to_dict()}")

        assert self.X.shape[0] == self.y.shape[0]
        assert self.X.shape[0] == self.y_encoded.shape[0]
        assert self.X.shape[0] == self.groups.shape[0]
        assert self.X.shape[1] == self.variant_info_df.shape[0]
        assert not pd.isna(self.groups).any(), f"'{self.group_label_name}' contains missing values"

    def drop_notusing_sample(self, notusing_list):
        indices_to_drop = self.label_df[self.label_df[self.target_label_name].isin(notusing_list)].index

        if not indices_to_drop.empty:
            self.label_df = self.label_df.drop(indices_to_drop)
            self.y = self.y.drop(indices_to_drop)
            self.X = np.delete(self.X, indices_to_drop, axis=0)

        logger.info(f"[progress] Dropped {len(indices_to_drop)} samples from the dataset. X.shape: {self.X.shape}, y.shape: {self.y.shape}")

    def encode_y(self):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(self.y)
        label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

        return y_encoded, label_mapping

    def get_data(self):
        return (self.X, np.array(self.y), self.y_encoded), self.groups, self.label_mapping, self.variant_info_df
        
    def get_combined_df(self):
        df = pd.DataFrame(self.X)
        new_columns = ['com' + str(i) for i in range(1, len(df.columns) + 1)]
        df.columns = new_columns
        df['country_encoded'] = self.y_encoded

        return df




# def export_feature_array(gt_array, output_file_prefix):
#     numpy_save_file_name = f"{output_file_prefix}_feature.npy"

#     gt_array_flatten = gt_array.reshape(gt_array.shape[0], -1) #flatten last feature dims

#     np.save(numpy_save_file_name, gt_array_flatten)
#     print(f"numpy array of shape (#samples, #features) : {gt_array_flatten.shape} -> saved to {numpy_save_file_name}. Original shape was {gt_array.shape}")


# def save_numpy_array(gt_array, numpy_save_file_name):
#     np.save(numpy_save_file_name, gt_array)
#     print(f"numpy array of shape (#samples, #features) : {gt_array.shape} -> saved to {numpy_save_file_name}")


def save_preprocessed_data(gt_array, variant_info_df, output_file_prefix):
    numpy_save_file_name = f"{output_file_prefix}_matrix.npy"
    pandas_save_file_name = f"{output_file_prefix}_variant.csv"

    np.save(numpy_save_file_name, gt_array)
    variant_info_df.to_csv(pandas_save_file_name, sep=",", index = False)

    print(f"genotype matrix shape (#samples, #features) : {gt_array.shape} -> saved to {numpy_save_file_name}")
    print(f"variant info dataframe (#features, ): {variant_info_df.shape} -> saved to {pandas_save_file_name}")


def read_preprocessed_data(target_file_prefix):
    mat_file_name = f"{target_file_prefix}_matrix.npy"
    variant_info_file_name = f"{target_file_prefix}_variant.csv"
    print(f"Reading data from files {mat_file_name} and {variant_info_file_name}")

    if (not os.path.isfile(mat_file_name)) or (not os.path.isfile(variant_info_file_name)):
        raise Exception(f"can not find preprocessed files starting with {target_file_prefix}")

    gt_array = np.load(mat_file_name)
    variant_info_df = pd.read_csv(variant_info_file_name, dtype = str)

    assert gt_array.shape[1] == variant_info_df.shape[0]

    print(f"Read genotype array of shape {gt_array.shape} and variant info dataframe of shape {variant_info_df.shape}")
    return gt_array, variant_info_df


class SNPDataSet:
    def __init__(self, genotype_array, variant_info_df, sample_metadata_df):
        """
        Initialize the SNPDataSet with genotype data, variant information, and sample metadata.

        :param genotype_array: NumPy array with shape (# samples, # snps)
        :param variant_info_df: Pandas DataFrame with shape (# snps, variant information)
        :param sample_metadata_df: Pandas DataFrame with shape (# samples, sample metadata)
        """

        assert genotype_array.shape[0] == sample_metadata_df.shape[0]
        assert genotype_array.shape[1] == variant_info_df.shape[0]

        self.genotype_array = genotype_array
        self.variant_info_df = variant_info_df
        self.sample_metadata_df = sample_metadata_df


    @classmethod
    def from_file(cls, target_file_prefix, sample_metadata_df):
        """
        Initialize the SNPDataSet from files.

        :param target_file_prefix: Prefix for file saving genotype array ("{target_file_prefix}_matrix.npy") and variant information ("{target_file_prefix}__variant.csv")
        :param sample_metadata_df: sample metadata dataframe
        """

        genotype_array, variant_info_df = read_preprocessed_data(target_file_prefix)

        # Create an instance of SNPDataSet with the loaded data
        return cls(genotype_array, variant_info_df, sample_metadata_df)
    
    def filter_variant(self, filter_array, inplace = True):
        """Fileter variants by the given boolean array"""

        assert filter_array.shape[0] == self.genotype_array.shape[1], f"Filter size is not matched. Passed filter shape: {filter_array.shape}, Genotype array: {self.genotype_array.shape}"

        if inplace:
            num_variant_before = self.genotype_array.shape[1]

            self.genotype_array = self.genotype_array[:, filter_array]
            self.variant_info_df = self.variant_info_df.iloc[filter_array]
            if hasattr(self, 'genotype_array_onehot') and self.genotype_array_onehot is not None:
                self.genotype_array_onehot = self.genotype_array_onehot[:, filter_array]

            assert self.genotype_array.shape[1] == self.variant_info_df.shape[0] 
            num_variant_after = self.genotype_array.shape[1]

            print(f"Filter variant retained {num_variant_after} / {num_variant_before}")
        else:
            num_variant_before = self.genotype_array.shape[1]

            new_snp_dataset = SNPDataSet(self.genotype_array[:, filter_array], self.variant_info_df.iloc[filter_array], self.sample_metadata_df)

            num_variant_after = new_snp_dataset.genotype_array.shape[1]

            print(f"Filter variant retained {num_variant_after} / {num_variant_before}")
            return(new_snp_dataset)

    def save_data(self, output_file_prefix):
        save_preprocessed_data(self.genotype_array, self.variant_info_df, output_file_prefix)

    # def create_onehot_genotype_array(self, inplace = False, batch_initialize = False):
    #     """
    #     Convert the genotype array into a one-hot encoded format.
    #     :input: self.genotype_array NumPy array with shape (# samples, # snps) with values 0 to 3
    #     :result: saved in self.genotype_array_onehot (# samples, # snps, 2) if inplace = True, else returned.
    #     """
    #     # Initialize an array of zeros with the new shape

    #     converted_array = np.empty((self.genotype_array.shape[0], self.genotype_array.shape[1], 2), dtype=np.int8)
    #     converted_array.fill(0)

    #     if batch_initialize:
    #         # perform batch by batch initialize to reduce memory consumption
    #         for i in tqdm(range(self.genotype_array.shape[0])):
    #             converted_array[i][(self.genotype_array[i] == 0)] = np.array([0, 0], dtype=np.int8)
    #             converted_array[i][(self.genotype_array[i] == 1)] = np.array([0, 1], dtype=np.int8)
    #             converted_array[i][(self.genotype_array[i] == 2)] = np.array([1, 0], dtype=np.int8)
    #             converted_array[i][(self.genotype_array[i] == 3)] = np.array([1, 1], dtype=np.int8)

    #     else:
    #         converted_array[np.where(self.genotype_array == 0)] = [0, 0]
    #         converted_array[np.where(self.genotype_array == 1)] = [0, 1]
    #         converted_array[np.where(self.genotype_array == 2)] = [1, 0]
    #         converted_array[np.where(self.genotype_array == 3)] = [1, 1]

    #     if inplace:
    #         self.genotype_array_onehot = converted_array
    #     else:
    #         return(converted_array)
            
    # def load_onehot_genotype_array(self, target_file_prefix):
    #     mat_file_name = f"{target_file_prefix}_matrix_onehot.npy"
    #     print(f"Reading one hot genotype array from file {mat_file_name}")

    #     if not os.path.isfile(mat_file_name):
    #         logging.warning(f"can not find preprocessed files starting with {target_file_prefix}")
    #         return None, None;

    #     self.genotype_array_onehot = np.load(mat_file_name)

    #     assert self.genotype_array.shape[1] == self.genotype_array_onehot.shape[1]


from datasets import load_dataset
from pathlib import Path
import pandas as pd 

class DatasetLoader:

    def __init__(self) -> None:
        pass

    def load_dataset(self, local:bool = True, **kwargs):
        """
        Load dataset from local or huggingface
        :param local: If True, load from local, else load from huggingface

        Optional kwargs:
        If local is True:
            path: Path to the dataset
            dataset_type: Type of dataset, csv or json
            load_as_pandas: If True, load as pandas dataframe, else load as huggingface dataset
        If local is False:
            dataset_name: Name of the dataset from huggingface datasets
            split: Split of the dataset to load, train, test, validation Default: train
            
        """
        dataset_name = kwargs.get("dataset_name", None)
        if local:
            dataset_type = kwargs.pop("dataset_type", "csv")
            if dataset_type == "csv":
                return self.load_local_csv(**kwargs)
            elif dataset_type == "json":
                return self.load_local_json(**kwargs)
            else:
                raise ValueError(f"Dataset type {dataset_type} not supported")
        else:
            split = kwargs.get("split", "train")
            return self.load_huggingface_dataset(**kwargs)
    
    def load_local_csv(self, **kwargs):
        """
        Load dataset from local csv file
        :param path: Path to the csv file
        :param load_as_pandas: If True, load as pandas dataframe, else load as huggingface dataset
        """
        path = kwargs.pop("path", None)
        load_as_pandas = kwargs.pop("load_as_pandas", False)
        data_type = "csv"
        if path is not None:
            if load_as_pandas:
                return pd.read_csv(path, **kwargs)
            return self.load_as_dataset(path, data_type)
        else:
            raise ValueError("Path not provided")
    
    def load_local_json(self, **kwargs):
        """
        Load dataset from local json file
        :param path: Path to the json file
        :param load_as_pandas: If True, load as pandas dataframe, else load as huggingface dataset
        """
        path = kwargs.pop("path", None)
        load_as_pandas = kwargs.pop("load_as_pandas", False)
        data_type = "json"
        if path is not None:
            if load_as_pandas:
                return pd.read_json(path, **kwargs)
            return self.load_as_dataset(path, data_type)
        else:
            raise ValueError("Path not provided")

    def load_as_dataset(self, path, data_type):
        """
        Load dataset from local file
        :param path: Path to the file
        :param data_type: Type of dataset, csv or json
        """
        if Path(path).exists():    
            return load_dataset(data_type, data_files=path)
        else:
            exception = FileNotFoundError(f"File {path} does not exist")
        #    logger.exception(exception)
            raise exception

    def load_huggingface_dataset(self, **kwargs):
        """
        Load dataset from huggingface datasets
        :param dataset_name: Name of the dataset from huggingface datasets
        :param split: Split of the dataset to load, train, test, validation Default: train
        """
        dataset_name = kwargs.pop("dataset_name", None)
        split = kwargs.pop("split", "train")

        if dataset_name is not None:
            return load_dataset(dataset_name, split=split, **kwargs)
        else:
            raise ValueError("Dataset name not provided")

    def push_dataset_to_huggingface_model_hub(self, dataset_name, dataset, **kwargs):
        """
        Push dataset to huggingface model hub
        :param dataset_name: Name of the dataset to push
        """
        dataset.push_to_hub(dataset_name, **kwargs)
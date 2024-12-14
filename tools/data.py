# this file has been written by meghana
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class AdultIncomeDataProcessor:
    def __init__(self):
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        self.column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num", 
            "marital-status", "occupation", "relationship", "race", "sex", 
            "capital-gain", "capital-loss", "hours-per-week", "native-country", 
            "income"
        ]
        self.preprocessor = None
        self.data = None

    def fetch_data(self):
        self.data = pd.read_csv(
            self.url, names=self.column_names, na_values=" ?", skipinitialspace=True
        )
        self.data.dropna(inplace=True)
    
    def preprocess_target(self):
        self.data["income"] = self.data["income"].apply(
            lambda x: 1 if x == ">50K" else 0
        ).astype(int)
    
    def create_preprocessor(self):
        categorical_cols = self.data.select_dtypes(include=["object", "category"]).columns
        numerical_cols = self.data.select_dtypes(include=["int64", "float64"]).columns.drop("income")
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(), categorical_cols),
            ]
        )
        self.preprocessor.fit(self.data.drop("income", axis=1))

    def get_input_dimension(self, X):
        X_transformed = self.preprocessor.transform(X)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        return X_transformed.shape[1]

    def process_data(self):
        self.fetch_data()
        self.preprocess_target()
        self.create_preprocessor()
        return self.data
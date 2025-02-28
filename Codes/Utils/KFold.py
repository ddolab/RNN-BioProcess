# K-fold
import numpy as np
import pandas as pd
from typing import Any, Iterator
from sklearn.model_selection import StratifiedKFold

class KFoldsCrossValidator(StratifiedKFold):
    """
    K-Folds cross validator returning stratified folds.
    """
    def __init__(
            self,
            data: pd.DataFrame,
            id_col: str,
            label_col: str,
            n_splits: int=5, 
            shuffle: bool=False, 
            random_state: int=None
    ) -> None:
        self.data = data.copy()
        self.ic_col = id_col
        self.label_column = label_col
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)

    def split(self) -> Iterator[Any]:
        """
        spliting K-Folds trai and test sets.
        """
        df = self.data
        X, y = [], []
        id_column = self.ic_col
        # get data into run id list and the corresponding lable list
        for group_name, group in df.groupby(by=id_column):
            X.append(group_name)
            y.append(group[self.label_column].values[0])
        X, y = np.array(X), np.array(y)

        for train_index, test_index in super().split(X, y):
            train_ids, test_ids = X[train_index], X[test_index]
            
            yield (
                df[df[id_column].isin(train_ids)], 
                df[df[id_column].isin(test_ids)]
            )
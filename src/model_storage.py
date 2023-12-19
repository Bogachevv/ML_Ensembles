from py_singleton import singleton
from pydantic import BaseModel
from typing import Optional, Union, List, Tuple, Any, Literal
import pandas as pd


class ModelRecord(BaseModel):
    model: Any
    target: str
    test_size: Optional[float] = None
    meta_info: Any = pd.Series()
    features: List[str] = list()
    status: Literal['not_fit', 'fit'] = 'not_fit'
    train_score: dict = dict()
    fit_curve: tuple = ()


@singleton
class Models(object):
    def __init__(self):
        self.models = dict()

    def __getitem__(self, item: int) -> ModelRecord:
        return self.models[item]

    def __setitem__(self, key: int, value: ModelRecord):
        self.models[key] = value

    def keys(self):
        return self.models.keys()

    def __contains__(self, item: int):
        return item in self.models

    def next_model_no(self) -> int:
        if self.models.keys():
            return max(self.models.keys()) + 1
        return 0

from src.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

from src.model.baseline import BaselineModel

MODEL_REGISTRY.register(BaselineModel)

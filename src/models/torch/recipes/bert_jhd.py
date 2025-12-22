# src/models/torch/recipes/bert_jhd.py
from src.models.torch.recipes.registry import register_torch_recipe
from src.models.torch.recipes.bert_jhd_recipe import BertJHDRecipe

@register_torch_recipe("bert_jhd_regression")
def build(cfg):
    return BertJHDRecipe(cfg)

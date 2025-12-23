# src/bootstrap.py
def bootstrap_registries() -> None:
    # Problems / Pipelines / Transforms
    import src.problems  # noqa: F401
    import src.data.pipelines.registry  # noqa
    import src.data.transforms  # noqa: F401

    # Engines
    import src.engines.torch.registry  # noqa: F401
    import src.engines.recbole.registry  # noqa F401
    import src.engines.sklearn.registry  # noqa: F401

    # Recipes
    import src.models.torch.recipes.registry  # noqa: F401
    import src.models.sklearn.recipes.registry  # noqa: F401
    import src.models.recbole.recipes.registry  # noqa: F401

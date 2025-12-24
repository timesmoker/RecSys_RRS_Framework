# src/bootstrap.py
def bootstrap_registries() -> None:
    # Problems / Pipelines / Transforms
    from src.problems import registry as problem_registry
    from src.data.pipelines import registry as pipeline_registry
    from src.data.transforms import registry as transform_registry

    problem_registry.bootstrap_problems()
    pipeline_registry.bootstrap_pipelines()
    transform_registry.bootstrap_transforms()

    # Engines
    import src.engines.torch.registry  # noqa: F401
    import src.engines.recbole.registry  # noqa F401
    import src.engines.sklearn.registry  # noqa: F401

    # Recipes
    from src.models.torch.recipes import registry as torch_recipe_registry
    from src.models.sklearn.recipes import registry as sklearn_recipe_registry
    from src.models.recbole.recipes import registry as recbole_recipe_registry

    torch_recipe_registry.bootstrap_torch_recipes()
    sklearn_recipe_registry.bootstrap_sklearn_recipes()
    recbole_recipe_registry.bootstrap_recbole_recipes()

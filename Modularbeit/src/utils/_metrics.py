def is_in_pipeline(model, model_class):
    return any(isinstance(step, model_class) for _, step in model.steps)
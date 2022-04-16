

MODEL_TYPES = [
    'Logistic',
    'RandomForest',
    'XGBoost'
]


def run_model_build(model_type="Logistic"):

    assert(model_type in MODEL_TYPES)

    sk_model = None

    return sk_model
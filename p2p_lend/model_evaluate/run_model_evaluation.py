

EVALUATION_TYPES = [
    'test_performance',
    'explainability'
]


def run_model_evaluation(model, output_path,use_dummy=None, evaluation_type="explainability"):

    assert(evaluation_type in EVALUATION_TYPES)

    evaluation = None

    return evaluation

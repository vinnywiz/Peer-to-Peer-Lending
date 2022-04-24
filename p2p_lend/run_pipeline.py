from p2p_lend.etl import run_etl
from p2p_lend.model_build import run_model_build
from p2p_lend.model_evaluate import run_model_evaluation
import logging

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

def run_pipeline(dummy=False, apply_smote=True, model_type='XGBoost',data_path='data', output_path='results'):

    if dummy:
        logger.info("Running ETL with dummy data")
        run_etl(dummy=True, data_path=data_path)
    else:
        logger.info("Running ETL with real data")
        run_etl(dummy=False, data_path=data_path)

    logger.info("Building Model")
    model = run_model_build(use_dummy= dummy,apply_smote=apply_smote, model_type=model_type)
    logger.info("Model build complete")

    logger.info("Evaluating model performance")
    run_model_evaluation(data_path=data_path, use_dummy=dummy, output_path=output_path, model_type=model_type, apply_smote=True)
    logger.info(f'Model evaluation output to path={output_path}')
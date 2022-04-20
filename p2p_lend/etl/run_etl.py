import logging
from .preprocess import preprocess_data
#     join_data,
#     smote_data,
#     feature_engineer
# )

logger = logging.getLogger(__name__)


def run_etl(dummy=True, data_path='data'):

    if dummy:
        logger.info(f"Preprocessing loan and listing data from {data_path}/dummy_data")
        list_df, loan_df, cleaned_df = preprocess_data(use_dummy=True, data_path='data')
        logger.info("Preprocessing successful!")
    else:
        logger.info(f"Preprocessing loan and listing data from {data_path}/real_data")
        list_df, loan_df, cleaned_df = preprocess_data(use_dummy=False, data_path='data')
        logger.info("Preprocessing successful!")

    return 'Success'
    # logger.info("Joining loan data to listings")
    # join_data(data_path=data_path)
    #
    # logger.info("Interpolating data with SMOTE")
    # smote_data(data_path=data_path)
    #
    # logger.info("Running feature engineering to get model ready data")
    # feature_engineer(data_path=data_path)
- [Peer to Peer Lending ML application](#peer-to-peer-lending-ml-application)
    - [How to get started:](#how-to-get-started)
      - [Dummy Data](#dummy-data)
      - [Dummy Data - Skip to Step 9](#dummy-data---skip-to-step-9)
      - [Real Data](#real-data)
      - [Results](#results)

# Peer to Peer Lending ML application
- The application is build using Prosper data
- Due to restrictions of uploading actual Prosper data. The application is run using randomly regerated dummy data. 

### How to get started:
1. Download the code repository
2. Ensure python 3 is installed on you local machine with pip
3. There are certain python packages that are necessary for the application to run. You can install through the below command in powershell.
'''
    cd Peer-to-Peer-Lending
    pip install -r requirements.txt
'''
4. Run the below command to install the application packages
```shell
    pip install -e .
```
#### Dummy Data 
5. Dummy Data
- By default the application is set to be run on dummy data. Run the below command to execute the application
```python
    from p2p_lend.run_pipeline import run_pipeline
    run_pipeline()
```
#### Dummy Data - Skip to Step 9

#### Real Data
6. Real Data
- The real data needs to be downloaded from Prosper Website. 
    - If you’d like to download Prosper listings history in CSV format, you can download this information at the following location(You must be logged in to Prosper):
        - https://www.prosper.com/investor/marketplace#/download
- The below folder structure needs to be created in the data folder
    - data
        - real_data
            - analysis
            - listings
            - loans
            - model
            - processed_data
    - The listings data files should be saved as
        - From: Listings_20050101to20130101_20210114T163603.zip
        - To: Listings_2005.zip
    - The loans data files should be saved as
        - From: Loans_20130101to20140101_20220422T060016.zip
        - To: Loans_2013.zip
    - Modify YAML config file
        - p2p_lend\etl\column_config.yaml
        - Set year parameters to which years of data you want to process using the application. Default is set to 2015 to 2021
        - year_start: 2015
        - year_end: 2021

7. Updates to application
- p2p_lend/run_pipeline.py
    - Before update - The dummy parameter is set to True by default
        - def run_pipeline(dummy=True, apply_smote=True, model_type='XGBoost',data_path='data', output_path='results'):
    - After update - The dummy parameter needs to be updated to False
        - def run_pipeline(dummy=False, apply_smote=True, model_type='XGBoost',data_path='data', output_path='results'):

8. Run updated code.
```shell
    pip install -e .
```

```python
    from p2p_lend.run_pipeline import run_pipeline
    run_pipeline()
```

#### Results
9. Final Results
- Results
    - classification_report.png
        - Shows a heat map of the classification model on a test dataset
    - global_bar_plot.png
        - Shows global feature importance
    - global_bee_plot.png
        - Shows global feature importance on model output as beeswarm plot 
    - Local_Explanability_Listing_Id_{0-4}.png
        - Shows local explanability of 5 listings
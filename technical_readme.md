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
5. Dummy Data
- By default the application is set to be run on dummy data. Run the below command to execute the application
```python
    from p2p_lend.run_pipeline import run_pipeline
    run_pipeline()
```
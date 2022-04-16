import pandas as pd
import os

def preprocess_data(data_path):




def compress_process_prosper_listings(source_path, destination_path):
    list_of_files = os.listdir(source_path)
    files_with_size = [(file_name, os.stat(os.path.join(source_path, file_name)).st_size) for file_name in
                       list_of_files]
    for file_to_read, size in files_with_size:
        # Only processing the files which have the term 'Listing' in the name
        # if 'Listing' in file_to_read:
        if 'Listings' in file_to_read:
            file = file_to_read[:13]
            file_name = file + '.csv'
            file_name1 = file + '_1.csv'
            file_name2 = file + '_2.csv'
            zip_path = destination_path + file + '.zip'
            zip_path1 = destination_path + file + '_1.zip'
            zip_path2 = destination_path + file + '_2.zip'
            read_path = source_path + file_to_read
            df = pd.read_csv(read_path, encoding_errors='ignore', low_memory=False)
            df_trimmed = df
            print(df_trimmed.shape)
            # splitting the dataframe into multiple files it exceed a certain original file size.
            # The number of files should be automated. For now it is just going to be split into 2 files if the original file exceeds a certain size
            num_files = math.ceil(size / 90000000)
            if num_files == 1:
                df_trimmed.to_csv(zip_path, compression={'method': 'zip', 'archive_name': file_name}, index=False)
            else:
                df_trimmed[:186000].to_csv(zip_path1, compression={'method': 'zip', 'archive_name': file_name1},
                                           index=False)
                df_trimmed[186000:].to_csv(zip_path2, compression={'method': 'zip', 'archive_name': file_name2},
                                           index=False)
    return 'Success'
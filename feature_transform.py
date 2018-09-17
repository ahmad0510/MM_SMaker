#!/usr/bin/env python

import sys
import os
import io
import pandas as pd
import boto3

#############################Include any necessary libraries here (IMPORTS)###############################################

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')

def feature_engineering(mode, pretrain_data_location, readytoscore_data_location, training_set_location, scoring_set_location):
    
    def pd_read_csv_s3(path, *args, **kwargs):
        path = path.replace("s3://", "")
        bucket, key = path.split('/', 1)
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj['Body'].read()), *args, **kwargs)

    #pd_read_csv_s3(pretrain_data_location, skiprows=2)
    #pd_read_csv_s3(pretrain_data_location)
    
    if mode == "train":
        input_data = pd_read_csv_s3(pretrain_data_location)
    elif mode == "score":
        input_data = pd_read_csv_s3(readytoscore_data_location) 
        
    ##########################feature engineering code begins here (EDITABLE)###################################
    
    ## example of removing the last two columns
    

    
    output_data = input_data.iloc[:,:-2]
    




    
    ##########################feature engineering code ends here (EDITABLE)#####################################
    
    if os.path.exists("output_data_temp.csv"):
        os.remove("output_data_temp.csv")
    
    output_data.to_csv("output_data_temp.csv", index = False)
    
    training_set_location = training_set_location.replace("s3://", "")
    scoring_set_location = scoring_set_location.replace("s3://", "")
    
    training_bucket, training_set_key = training_set_location.split('/', 1)
    scoring_bucket, scoring_set_key = scoring_set_location.split('/', 1)
    
    
    if mode == "train":
        s3_resource.meta.client.upload_file('output_data_temp.csv', training_bucket, training_set_key)

    elif mode == "score":
        s3_resource.meta.client.upload_file('output_data_temp.csv', scoring_bucket, scoring_set_key)
        
#if __name__ == '__main__':
#    # Map command line arguments to function arguments.
#    feature_engineering(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] )
        
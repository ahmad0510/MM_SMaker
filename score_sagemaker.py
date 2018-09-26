#!/usr/bin/env python
import time
import boto3
import json
import numpy
import io
import os
import sys
import pandas as pd
sm = boto3.client('sagemaker')

#.................................Read the config files.........................


################## ARGUMENTS PASSED TO PYTHON CODE#########################
################## ONLY FOR TESTING. REMOVE BEFORE DEPLOYMENT #####################

BucketName = "agco-dev-dl1"
UseCaseName = "iris-test"
ModelName = "germany"
ModelVersion = "v-1"

################## TEST CODE #########################################

## read the BucketName, UseCaseName, ModelName and ModelVersion passed as arguments to the scoring script
s3 = boto3.resource('s3')
content_object_info = s3.Object(BucketName, "scripts" + "/" + UseCaseName + "/" + ModelName + "-" + ModelVersion + "/" + "config_info.json")
content_object_paths = s3.Object(BucketName, "scripts" + "/" + UseCaseName + "/" + ModelName + "-" + ModelVersion + "/" + "config_paths.json")
content_object_train_job = s3.Object(BucketName, "scripts" + "/" + UseCaseName + "/" + ModelName + "-" + ModelVersion + "/" + "train_job_details.json")
content_object_model_params = s3.Object(BucketName, "scripts" + "/" + UseCaseName + "/" + ModelName + "-" + ModelVersion + "/" + "config_model_params.json")
content_object_instance_size = s3.Object(BucketName, "scripts" + "/" + UseCaseName + "/" + ModelName + "-" + ModelVersion + "/" + "instance_size.json")

json_content_info = json.loads(content_object_info.get()['Body'].read().decode('utf-8'))
json_content_paths = json.loads(content_object_paths.get()['Body'].read().decode('utf-8'))
json_content_train_job = json.loads(content_object_train_job.get()['Body'].read().decode('utf-8'))
json_content_model_params = json.loads(content_object_model_params.get()['Body'].read().decode('utf-8'))
json_content_instance_size = json.loads(content_object_instance_size.get()['Body'].read().decode('utf-8'))


config_params_model_scoring_mode = json_content_model_params["scoring_mode"]
use_case_name = json_content_info["use_case_name"]
model_name = json_content_info["model_name"]
model_version = json_content_info["model_version"]
bucket_name = json_content_paths["bucket_name"]
scores_file_base_path = json_content_paths["scores_file_base_path"]
scoring_instance = json_content_instance_size["scoring_instance"]
pretrain_data_base_path = json_content_paths["pretrain_data_base_path"]
pretrain_file_name = json_content_paths["pretrain_file_name"]
readytoscore_data_base_path = json_content_paths["readytoscore_data_base_path"]
readytoscore_file_name = json_content_paths["readytoscore_file_name"]
training_set_base_path = json_content_paths["training_set_base_path"]
training_set_file_name = json_content_paths["training_set_file_name"]
scoring_set_base_path = json_content_paths["scoring_set_base_path"]
scoring_set_file_name = json_content_paths["scoring_set_file_name"]
endpoint_config_name = json_content_train_job["endpoint_config_name"]
endpoint_name = json_content_train_job["endpoint_name"]


# this is the data that is input for feature engineering during training
pretrain_data_location = bucket_name + "/" + pretrain_data_base_path + "/" + use_case_name + "/" + pretrain_file_name

# this is the data that is input for feature engineering during scoring
readytoscore_data_location = bucket_name + "/" + readytoscore_data_base_path + "/" + use_case_name + "/" + readytoscore_file_name

# create the training and validation sets with the specified name format and place them in the correct folders
training_set_file_location = bucket_name + "/" + training_set_base_path + "/" + use_case_name + "/" + training_set_file_name
scoring_set_file_location = bucket_name + "/" + scoring_set_base_path + "/" + use_case_name + "/" + scoring_set_file_name

scoring_set_prefix = scoring_set_base_path + "/" + use_case_name

endpoints_all_json = sm.list_endpoints()
endpoints_list = endpoints_all_json['Endpoints']
endpoint_name  = json.loads(open('train_job_details.json').read())["endpoint_name"]
endpoint_exists = None
for ep in endpoints_list:
    if endpoint_name == ep['EndpointName']:
        endpoint_exists = True
        break
    else:
        endpoint_exists = False

if endpoint_exists == False:
    if config_params_model_scoring_mode == "batch":
        ## Read the endpoint config and names from the json file

        create_endpoint_response = sm.create_endpoint(EndpointName=endpoint_name,EndpointConfigName=endpoint_config_name)

        resp = sm.describe_endpoint(EndpointName=endpoint_name)
        status = resp['EndpointStatus']
        print("Status: " + status)

        try:
            sm.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
        finally:
            resp = sm.describe_endpoint(EndpointName=endpoint_name)
            status = resp['EndpointStatus']
            print("Arn: " + resp['EndpointArn'])
            print("Create endpoint ended with status: " + status)

            if status != 'InService':
                message = sm.describe_endpoint(EndpointName=endpoint_name)['FailureReason']
                print('Create endpoint failed with the following error: {}'.format(message))
                raise Exception('Endpoint creation did not succeed')
            

#..................Call the model transformation script ...................................
sys.path.insert(0, "s3://" + BucketName + "/" + "scripts" + "/" + UseCaseName + "/" + ModelName + "-" + ModelVersion)
import feature_transform
#import "s3://" + BucketName + "/" + UseCaseName + "/" + ModelName + "-" + ModelVersion + "/" + "feature_transform"


feature_transform.feature_engineering("score", pretrain_data_location, readytoscore_data_location, training_set_file_location, scoring_set_file_location)

#................. Scoring process..........................................................

runtime = boto3.Session().client('sagemaker-runtime')
s3_resource = boto3.resource('s3')

# Simple function to create a csv from our numpy array
def np2csv(arr):
    csv = io.BytesIO()
    numpy.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()


payload_df = pd.read_csv("{}/{}/{}".format(bucket_name,scoring_set_prefix, scoring_set_file_name))
payload_np = payload_df.values
payload = np2csv(payload_np)

response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='text/csv', Body=payload)

result = response['Body'].read().decode('ascii')

if os.path.exists("output_predictions.txt"):
    os.remove("output_predictions.txt")
    with open("output_predictions.txt", "w") as text_file:
        text_file.write(result + "\n")
else:
    with open("output_predictions.txt", "w") as text_file:
        text_file.write(result + "\n")

s3_resource.meta.client.upload_file('output_predictions.txt', bucket_name[5:], scores_file_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "-" + "predictions.txt")

#....................Perform cleanup............................................................

## Clean the endpoints in case of batch scoring process
if config_params_model_scoring_mode == "batch":
    sm.delete_endpoint(EndpointName=endpoint_name)

## Delete the temporary files
if os.path.exists("output_predictions.txt"):
    os.remove("output_predictions.txt")  

if os.path.exists("output_data_temp.csv"):
    os.remove("output_data_temp.csv")

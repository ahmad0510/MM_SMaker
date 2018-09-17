#!/usr/bin/env python

import boto3
from time import gmtime, strftime
import pandas as pd
import os
use_case_name = json.loads(open(r'config_info.json').read())["use_case_name"]
model_name = json.loads(open(r'config_info.json').read())["model_name"]
model_version = json.loads(open(r'config_info.json').read())["model_version"]
bucket_name = json.loads(open(r'config_paths.json').read())["bucket_name"]
scores_file_base_path = json.loads(open(r'config_paths.json').read())["scores_file_base_path"]
scoring_instance = json.loads(open(r'instance_size.json').read())["scoring_instance"]
readytoscore_data_base_path = json.loads(open(r'config_paths.json').read())["readytoscore_data_base_path"]
readytoscore_file_name = json.loads(open(r'config_paths.json').read())["readytoscore_file_name"]
scoring_set_base_path = json.loads(open(r'config_paths.json').read())["scoring_set_base_path"]
scoring_set_file_name = json.loads(open(r'config_paths.json').read())["scoring_set_file_name"]
model_artifacts_base_path = json.loads(open(r'config_paths.json').read())["model_artifacts_base_path"]
train_model_name = json.loads(open(r'train_job_details.json').read())["train_model_name"]
train_job_Arn = json.loads(open(r'train_job_details.json').read())["train_job_Arn"]
model_artifact_path = json.loads(open(r'train_job_details.json').read())["model_artifact_path"]
output_scores_prefix = scores_file_base_path + "/" + use_case_name
scoring_set_prefix = scoring_set_base_path + "/" + use_case_name
scoring_set_df = pd.read_csv("{}/{}/{}".format(bucket_name,scoring_set_prefix, scoring_set_file_name))


s3 = boto3.resource('s3')
sm = boto3.client('sagemaker')
endpoints_all_json = sm.list_endpoints()
endpoints_list = endpoints_all_json['Endpoints']
endpoint_name  = json.loads(open(r'train_job_details.json').read())["endpoint_name"]
endpoint_exists = None
for ep in endpoints_list:
    if endpoint_name == ep['EndpointName']:
        endpoint_exists = True
    else:
        endpoint_exists = False

if endpoint_exists:

else:
	from sagemaker.predictor import csv_serializer
	predictor = model_clf.deploy(1, scoring_instance, serializer=csv_serializer)


output_predictions = predictor.predict(scoring_set_df.values).decode('utf-8')
output_predictions_list = output_predictions.split('\n')
output_predictions_csv = pd.DataFrame(output_predictions_list)
output_predictions_csv.to_csv('output_predictions.csv', header = False, index = False)
s3.meta.client.upload_file('output_predictions.csv', bucket_name[5:], scores_file_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "-" + "predictions.csv")

#!/usr/bin/env python

import boto3
from time import gmtime, strftime

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
train_model_name = json.loads(open(r'train_job_name.json').read())["train_model_name"]
model_artifact_path = json.loads(open(r'train_job_name.json').read())["model_artifact_path"]


#model_artifact_path = bucket_name + "/" + model_artifacts_base_path
job_name = 'Batch-Transform-' + use_case_name + "-" + model_name + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
output_scores_prefix = scores_file_base_path + "/" + use_case_name
scoring_set_prefix = scoring_set_base_path + "/" + use_case_name

sm = boto3.client('sagemaker')

request = \
{
    "TransformJobName": job_name,
    "ModelName": train_model_name,
    "MaxConcurrentTransforms": 1,
    "MaxPayloadInMB": 10,
    "BatchStrategy": "SingleRecord",
    "TransformOutput": {
        "S3OutputPath": "{}/{}".format(bucket_name,output_scores_prefix)
    },
    "TransformInput": {
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": "{}/{}/{}".format(bucket_name,scoring_set_prefix, scoring_set_file_name),
            }
        }
    },
    "TransformResources": {
            "InstanceType": scoring_instance ,
            "InstanceCount": 1
    }
}
                            
response = sm.create_transform_job(**request)

response = sm.describe_transform_job(TransformJobName=job_name)

import boto3
from time import gmtime, strftime

request = \
{
   "StatusEquals": "Completed",
   "SortBy": "CreationTime",
   "SortOrder": "Descending",
   "MaxResults": 20,
}

response = sm.list_transform_jobs(**request)

#response = sm.stop_transform_job(TransformJobName=job_name)
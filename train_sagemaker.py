#!/usr/bin/env python

import os
import json
import boto3 
import re
import numpy as np
import pandas as pd
from sagemaker import get_execution_role
import sagemaker as sage
from time import gmtime, strftime

sess = sage.Session()
###################################............The role may need to be manually specified........#####################################
role = get_execution_role()
s3 = boto3.client("s3")


bucket_name = json.loads(open(r'config_paths.json').read())["bucket_name"]
pretrain_data_base_path = json.loads(open(r'config_paths.json').read())["pretrain_data_base_path"]
training_set_base_path = json.loads(open(r'config_paths.json').read())["training_set_base_path"]
readytoscore_data_base_path = json.loads(open(r'config_paths.json').read())["readytoscore_data_base_path"]
pretrain_file_name = json.loads(open(r'config_paths.json').read())["pretrain_file_name"]
readytoscore_file_name = json.loads(open(r'config_paths.json').read())["readytoscore_file_name"]
training_set_file_name = json.loads(open(r'config_paths.json').read())["training_set_file_name"]
scoring_set_base_path = json.loads(open(r'config_paths.json').read())["scoring_set_base_path"]
scoring_set_file_name = json.loads(open(r'config_paths.json').read())["scoring_set_file_name"]
model_artifacts_base_path = json.loads(open(r'config_paths.json').read())["model_artifacts_base_path"]


use_case_name = json.loads(open(r'config_info.json').read())["use_case_name"]
model_name = json.loads(open(r'config_info.json').read())["model_name"]
model_version = json.loads(open(r'config_info.json').read())["model_version"]

# this is the data that is input for feature engineering during training
pretrain_data_location = bucket_name + "/" + pretrain_data_base_path + "/" + use_case_name + "/" + pretrain_file_name

# this is the data that is input for feature engineering during scoring
readytoscore_data_location = bucket_name + "/" + readytoscore_data_base_path + "/" + use_case_name + "/" + readytoscore_file_name

# create the training and validation sets with the specified name format and place them in the correct folders
training_set_file_location = bucket_name + "/" + training_set_base_path + "/" + use_case_name + "/" + training_set_file_name
scoring_set_file_location = bucket_name + "/" + scoring_set_base_path + "/" + use_case_name + "/" + scoring_set_file_name

# training and scoring instance values
training_instance = json.loads(open(r'instance_size.json').read())["training_instance"]
scoring_instance = json.loads(open(r'instance_size.json').read())["scoring_instance"]
num_training_instances = int(json.loads(open(r'instance_size.json').read())["num_training_instances"])
num_scoring_instances = int(json.loads(open(r'instance_size.json').read())["num_scoring_instances"])


#..................Call the model transformation script ...................................

import feature_transform

feature_transform.feature_engineering("train", pretrain_data_location, readytoscore_data_location, training_set_file_location, scoring_set_file_location)

#..................Training job begins here ...............................................
# train the model
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
#image_placeholder = '{}.dkr.ecr.{}.amazonaws.com/' + use_case_name + ':latest'
image_placeholder = '{}.dkr.ecr.{}.amazonaws.com/' + use_case_name + ":" + model_name

image = image_placeholder.format(account, region)
model_artifact_folder = bucket_name + "/" + model_artifacts_base_path

# read the instance size from config file
# parametrize the model name as per use case, model name and model version
model_clf = sage.estimator.Estimator(image,
                       role, num_training_instances, training_instance,
                       output_path=model_artifact_folder,              
                       sagemaker_session=sess)

model_clf.fit(training_set_file_location)


#.................Training job complete ....................................................

#.................Create end point config and deploy the model ............................,

## create model

sm = boto3.client('sagemaker')
train_job_name = model_clf._current_job_name
info = sm.describe_training_job(TrainingJobName=train_job_name)
train_job_Arn = info['TrainingJobArn']


model_data = info['ModelArtifacts']['S3ModelArtifacts']

primary_container = {
    'Image': image,
    'ModelDataUrl': model_data
}

create_model_response = sm.create_model(
    ModelName = train_job_name,
    ExecutionRoleArn = role,
    PrimaryContainer = primary_container)


## create endpoint configuration

endpoint_config_name = use_case_name +"-" + model_name + "-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':scoring_instance,
        'InitialInstanceCount':num_scoring_instances,
        'ModelName':train_job_name,
        'VariantName':'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

#...........Store the endpoint details................................................

endpoint_name = use_case_name +"-" + model_name + "-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

training_job_dict = {"train_job_Arn" : train_job_Arn, "train_job_name" : train_job_name, "model_artifact_path" : model_data, "endpoint_config_name" : endpoint_config_name, "endpoint_name" : endpoint_name }
with open('train_job_details.json', 'w') as outfile:  
    json.dump(training_job_dict, outfile)


#..........................Create the endpoints if scoring mode is real time ........................

config_params_model_scoring_mode = json.loads(open(r'config_model_params.json').read())["scoring_mode"]

if config_params_model_scoring_mode == "real":
    ## Read the endpoint config and names from the json file
    endpoint_config_name = json.loads(open(r'train_job_details.json').read())["endpoint_config_name"]
    endpoint_name = json.loads(open(r'train_job_details.json').read())["endpoint_name"]
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

    
#....................Move the config files into a dedicated location..........................

model_scripts_base_path = json.loads(open(r'config_paths.json').read())["model_scripts_base_path"]


s3_resource = boto3.resource('s3')
s3_resource.meta.client.upload_file('config_model_params.json', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "config_model_params.json")
s3_resource.meta.client.upload_file('config_info.json', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "config_info.json")
s3_resource.meta.client.upload_file('config_paths.json', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "config_paths.json")
s3_resource.meta.client.upload_file('config_packages.csv', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "config_packages.csv")
s3_resource.meta.client.upload_file('instance_size.json', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "instance_size.json")
s3_resource.meta.client.upload_file('feature_transform.py', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "feature_transform.py")
s3_resource.meta.client.upload_file('train_job_details.json', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "train_job_details.json")
s3_resource.meta.client.upload_file('train_template.txt', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "train_template.txt")
s3_resource.meta.client.upload_file('predictor_template.txt', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "predictor_template.txt")
s3_resource.meta.client.upload_file('dockerfile_template.txt', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "dockerfile_template.txt")
s3_resource.meta.client.upload_file('docker_build.sh', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "docker_build.sh")
s3_resource.meta.client.upload_file('train_sagemaker.py', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "train_sagemaker.py")
s3_resource.meta.client.upload_file('score_sagemaker.py', bucket_name[5:], model_scripts_base_path + "/" + use_case_name + "/" + model_name + "-" + model_version + "/" + "score_sagemaker.py")
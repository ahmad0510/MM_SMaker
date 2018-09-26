#!/usr/bin/env python
import json
import os

# read config_info file
use_case_name = json.loads(open('config_info.json').read())["use_case_name"]
model_name = json.loads(open('config_info.json').read())["model_name"]
model_version = json.loads(open('config_info.json').read())["model_version"]

# read config_model_params file
config_params_kernel = json.loads(open('config_model_params.json').read())["kernel"]
config_params_model_library = json.loads(open('config_model_params.json').read())["model_library"]
config_params_model_family = json.loads(open('config_model_params.json').read())["model_family"]
config_params_model_type = json.loads(open('config_model_params.json').read())["model_type"]
config_params_model_hyperparameters = json.loads(open('config_model_params.json').read())["hyperparameters"]

# read the config_packages file
with open('config_packages.csv','r') as text_file:
    package_list = text_file.read().split(',')

# dynamically generate the train file
path = "container/ml_model/train"
if os.path.exists(path):
    os.remove(path)

if not os.path.exists(path):
    os.mknod(path)

model_library_text = "from "+ config_params_model_library + " import " + config_params_model_family

model_call_parameters = ""
for name, value in config_params_model_hyperparameters.items():
    model_call_parameters = model_call_parameters + name + "=" + value + ","
model_call_parameters = model_call_parameters[:-1]
model_call_text = "    " + "    " + "clf=" + config_params_model_family + "." + config_params_model_type + "(" + model_call_parameters + ")"
pickle_name_text = "    " + "    " + "with open(os.path.join(model_path," + "'" + use_case_name + "-" + model_name + "-" + model_version + ".pkl'" + "), 'w') as out:"
    
with open("train_template.txt") as f1:
    with open(path, 'a') as f2:
        lines = f1.readlines()
        i = 0
        while 0<=i<=15:
            f2.write(lines[i])
            i=i+1
        f2.write(model_library_text + "\n")
        i = 18
        while 18<=i<=51:
            f2.write(lines[i])
            i=i+1
        f2.write(model_call_text + "\n")
        i = 53
        while 53<=i<=55:
            f2.write(lines[i])
            i=i+1
        f2.write(pickle_name_text + "\n")
        i = 57
        while 57<=i<=74:
            f2.write(lines[i])
            i=i+1

# dynamically generate the predictor.py file
path = "container/ml_model/predictor.py"
if os.path.exists(path):
    os.remove(path)

if not os.path.exists(path):
    os.mknod(path)

pickle_call_text = "    " + "    " + "    "+ "with open(os.path.join(model_path, " +  "'" + use_case_name + "-" + model_name + "-" + model_version + ".pkl'" + "), 'r') as inp:"
    
with open("predictor_template.txt") as f1:
    with open(path, 'a') as f2:
        lines = f1.readlines()
        i = 0
        while 0<=i<=29:
            f2.write(lines[i])
            i=i+1
        f2.write(pickle_call_text + "\n")
        i = 31
        while 31<=i<=82:
            f2.write(lines[i])
            i=i+1

# dynamically generate the dockerfile
path = "container/Dockerfile"
if os.path.exists(path):
    os.remove(path)

if not os.path.exists(path):
    os.mknod(path)

package_import_text = "    " + "pip install "    
for package in package_list:
    package_import_text = package_import_text + " " + package 

package_import_text = package_import_text + " && \ "

with open("dockerfile_template.txt") as f1:
    with open(path, 'a') as f2:
        lines = f1.readlines()
        i = 0
        while 0<=i<=21:
            f2.write(lines[i])
            i=i+1
        f2.write(package_import_text + "\n")
        i = 22
        while 22<=i<=38:
            f2.write(lines[i])
            i=i+1            

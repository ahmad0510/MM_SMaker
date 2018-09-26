#chmod +x docker_setup.py
#chmod +x feature_transform.py
#chmod +x train_sagemaker.py
#chmod +x score_sagemaker.py

python docker_setup.py
sh docker_build.sh
python train_sagemaker.py

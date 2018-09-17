chmod +x docker_setup.py
chmod +x feature_transform.py
chmod +x train_sagemaker.py
chmod +x score_sagemaker.py

./docker_setup.py
sh docker_build.sh
./train_sagemaker.py
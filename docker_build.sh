# The name of our use_case and model
use_case_name="$(sudo cat config_info.json | jq .'use_case_name')"
use_case_name=$(echo "$use_case_name" | tr -d '"')
model_name=$(sudo cat config_info.json | jq .'model_name')
model_name=$(echo "$model_name" | tr -d '"')

cd container

chmod +x ml_model/train
chmod +x ml_model/serve

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to eu-west-1 if none defined)
region=$(aws configure get region)
region=${region:-eu-west-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${use_case_name}:$model_name"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${use_case_name}"

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${use_case_name}"
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build  -t ${use_case_name} .
docker tag ${use_case_name} ${fullname}

docker push ${fullname}
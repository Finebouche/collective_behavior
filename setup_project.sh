# Activate conda environment
conda env create -f environment.yml
conda activate collective_env

# to ignore changes to wandb_api_key.txt
git update-index --assume-unchanged wandb_api_key.txt

echo "Environment setup complete, don't forget to add your wandb key to wandb_api_key.txt."

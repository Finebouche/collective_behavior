# Activate conda environment
conda env create -f environment.yml

# to ignore changes to wandb_api_key.txt
git update-index --assume-unchanged wandb_api_key.txt

echo "Environment setup complete, don't forget to add your wandb key to wandb_api_key.txt."

python -m ipykernel install --user --name=collective_env
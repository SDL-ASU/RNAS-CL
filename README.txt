# Installation
pip install -r requirements.txt

# Search
cd DAS/imageNetDA
python search.py --config configs/search_config.yaml

# Train 
cd DAS/imageNetDA
python train --config configs/train_config.yaml

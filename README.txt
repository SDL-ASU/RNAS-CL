# Installation
pip install -r requirements.txt

# Search
cd RNAS-CL/imageNetDA
python search.py --config configs/search_config.yaml

# Train 
cd RNAS-CL/imageNetDA
python train --config configs/train_config.yaml

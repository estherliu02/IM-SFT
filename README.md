1. git clone  https://github.com/estherliu02/IM-SFT.git
2. cd IM-SFT
3. 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda create -n sft python=3.10
conda activate sft
pip install -r requirements.txt
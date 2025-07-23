1. git clone  https://github.com/estherliu02/IM-SFT.git
2. cd IM-SFT
3. 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda create -n sft python=3.10
conda activate sft
pip install torch==2.1.2+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

pip uninstall flash-attn -y

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

git checkout v2.2.2

pip install ninja packaging wheel

export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

python setup.py install


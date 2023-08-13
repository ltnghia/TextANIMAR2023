conda create -y -n shrec23 python=3.8.17
conda activate shrec23
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python3 -m pip install -r requirements.txt
conda install -y -c conda-forge faiss-gpu faiss-cpu

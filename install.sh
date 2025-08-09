pip install datasets transformers torch loguru tqdm
mkdir -p /workspace/logs
apt-get update
apt-get install -y python3-dev vim
pip install accelerate
pip install bitsandbytes
pip install -U "flash-attn>=2.6.0" 
pip install deepspeed
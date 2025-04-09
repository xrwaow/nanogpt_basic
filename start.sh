pip install transformers datasets matplotlib protobuf sentencepiece
git clone https://github.com/xrwaow/nanogpt_basic.git
cd nanogpt_basic/
mkdir data
cd data
wget https://huggingface.co/datasets/IxrI/fineweb_1b_mistral-v0.3_tokenized/resolve/main/1000046592.pt
cd ..
mv config_a100.py config.py
python train.py
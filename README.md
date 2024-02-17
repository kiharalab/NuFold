# NuFold
A 3D RNA structure prediction method

## setup
* install conda environment
```
conda create -n nufold_P python=3.8
conda activate nufold_P
pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install ml-collections==0.1.0 dm-tree==0.1.6 deepspeed==0.5.3 protobuf==3.20.0 scipy==1.4.1 biopython==1.79
conda install numpy==1.21.0 -c conda-forge
```
* rMSA
`To be written`

* IPknot (Use precompiled binary)
```
wget https://github.com/satoken/ipknot/releases/download/v1.1.0/ipknot-1.1.0-x86_64-linux.zip
unzip ipknot-1.1.0-x86_64-linux.zip
rm ipknot-1.1.0-x86_64-linux/README.md
mv ipknot-1.1.0-x86_64-linux/ipknot /content
rmdir ipknot-1.1.0-x86_64-linux
rm ipknot-1.1.0-x86_64-linux.zip
chmod +x ipknot
```

* Download checkpoint
```
mkdir -p checkpoints
wget -O checkpoints/global_step145245.pt http://kiharalab.org/nufold/global_step145245.pt
```

## How to run
* run rMSA
```
rMSA/rMSA.pl
```

```
python3 run_nufold.py \
  --ckpt_path checkpoints/global_step146269.pt \
  --input_fasta data/input.fasta \
  --input_dir data/alignment/ \
  --output_dir output/ \
  --config_preset initial_training
```

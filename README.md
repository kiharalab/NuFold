# NuFold: 3D RNA Structure Prediction Method

NuFold is a state-of-the-art method designed for predicting 3D RNA structures, leveraging deep learning for high accuracy and reliability. This tool is particularly useful for biologists and bioinformatics researchers focusing on RNA function and structure.

## Environment Setup and Installation

### 1. Conda Environment

Start by setting up a dedicated Conda environment:

```bash
conda create -n nufold_P python=3.8
conda activate nufold_P
```

### 2. PyTorch and Related Libraries

Install PyTorch and associated libraries with CUDA support for optimized performance:

```bash
pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. Additional Dependencies

Install the necessary Python packages for NuFold:

```bash
pip3 install ml-collections==0.1.0 dm-tree==0.1.6 deepspeed==0.5.3 protobuf==3.20.0 scipy==1.4.1 biopython==1.79
conda install numpy==1.21.0 -c conda-forge
```

### 4. Aria2 for Downloading

For efficient downloading of large files, install Aria2:

```bash
apt-get install aria2
```

## rMSA Configuration

rMSA is crucial for preparing RNA sequences. Clone its repository and set up the database:

```bash
git clone https://github.com/pylelab/rMSA
cd rMSA/database/
aria2c -q -R -x 16 -j 20 -s 65536 -c --optimize-concurrent-downloads https://kiharalab.org/nufold/database.zip
unzip database.zip && rm database.zip
cd ../..
```

## IPknot Setup

IPknot is used for RNA secondary structure prediction. Download and set it up with the following commands:

```bash
wget https://github.com/satoken/ipknot/releases/download/v1.1.0/ipknot-1.1.0-x86_64-linux.zip
unzip ipknot-1.1.0-x86_64-linux.zip && rm ipknot-1.1.0-x86_64-linux.zip
chmod +x ipknot-1.1.0-x86_64-linux/ipknot
```

## Model Checkpoint

Download the NuFold model checkpoint to a designated directory:

```bash
mkdir -p checkpoints
wget -O checkpoints/global_step145245.pt http://kiharalab.org/nufold/global_step145245.pt
```

## Running NuFold

To predict RNA structures with NuFold, proceed as follows:

1. **Prepare RNA Sequences with rMSA:**

   Navigate to the rMSA directory and run the rMSA script.

   ```bash
   cd rMSA
   ./rMSA.pl
   ```

2. **Predict Structures with NuFold:**

   Use the provided script to run NuFold, adjusting paths as necessary.

   ```bash
   python3 run_nufold.py \
     --ckpt_path checkpoints/global_step146269.pt \
     --input_fasta data/input.fasta \
     --input_dir data/alignment/ \
     --output_dir output/ \
     --config_preset initial_training
   ```


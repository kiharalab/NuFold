# NuFold: 3D RNA Structure Prediction Method
![nufold](https://media.github.itap.purdue.edu/user/6911/files/5e300498-a159-40e4-9de4-a008ec7466fc)

NuFold is a state-of-the-art method designed for predicting 3D RNA structures, leveraging deep learning for high accuracy and reliability. This tool is particularly useful for biologists and bioinformatics researchers focusing on RNA function and structure.

License: GPL v3 for academic use. (For commercial use, please contact us for different licensing)
Contact: Daisuke Kihara (dkihara@purdue.edu)

Cite: [Kagaya, Y., Zhang, Z., Ibtehaz, N., Wang, X., Nakamura, T., Punuru, P.D., & Kihara, D. (2023). NuFold: A Novel Tertiary RNA Structure Prediction Method Using Deep Learning with Flexible Nucleobase Center Representation.](https://www.biorxiv.org/content/10.1101/2023.09.20.558715v1) In submission (an earlier version in bioRxiv).

Online Platform:
1. [Google Colab](https://colab.research.google.com/github/kiharalab/nufold/blob/master/ColabNuFold.ipynb)

## Environment Setup and Installation

### 1. Conda Environment
Start by setting up a dedicated Conda environment:

```bash
conda create -n nufold_P python=3.10
conda activate nufold_P
```

### 2. PyTorch and Related Libraries
Install the latest version of PyTorch and associated libraries with CUDA support for optimized performance:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### 3. Additional Dependencies
Install the necessary Python packages for NuFold:

```bash
pip install ml-collections dm-tree deepspeed protobuf scipy biopython numpy shutil
```

### 4. Aria2 for Downloading
For efficient downloading of large files, install Aria2:

```bash
apt-get install aria2
```

## rMSA Configuration
Clone rMSA and set up the database:

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

## Running NuFold with the End-to-End Script
We have created a new script that simplifies the process of running NuFold. To predict RNA structures with NuFold using the end-to-end script, follow these steps:

1. Make the `nufold.py` script executable:

```bash
chmod +x nufold.py
```

2. Run NuFold by providing the RNA sequence as a command-line argument:

```bash
./nufold.py your_sequence
```

The script will automatically generate a random job name, create the necessary directories, perform data preprocessing, run NuFold, and save the output in the `nufold_output.zip` file.
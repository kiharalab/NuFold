import os
import subprocess
import argparse
import shutil
import random
import string

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def preprocess_data(fasta_filepath, job_dir):
    run_command(f"ipknot {fasta_filepath} > {fasta_filepath.replace('.fasta', '')}.ipknot.ss")
    print("ipknot output saved")
    run_command(f"sed -i 's@$dbdir/nt@@g' rMSA/rMSA.pl")
    run_command(f"rMSA/rMSA.pl {fasta_filepath} -cpu=2")
    print("rMSA.pl run completed.")

def run_nufold(fasta_filepath, job_base_dir, output_dir):
    run_command(f"python run_nufold.py \
                  --ckpt_path checkpoints/global_step145245.pt \
                  --input_fasta {fasta_filepath} \
                  --input_dir {job_base_dir} \
                  --output_dir {output_dir} \
                  --config_preset initial_training")

def generate_job_name(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def main(rna_sequence):
    job_name = generate_job_name()
    output_dir = "nufold_output"

    jobs_dir = "jobs"
    os.makedirs(jobs_dir, exist_ok=True)
    job_base_dir = os.path.join(jobs_dir, f"{job_name}_base")
    os.makedirs(job_base_dir, exist_ok=True)
    job_dir = os.path.join(job_base_dir, job_name)
    os.makedirs(job_dir, exist_ok=True)

    fasta_filename = f"{job_name}.fasta"
    fasta_filepath = os.path.join(job_dir, fasta_filename)
    with open(fasta_filepath, "w") as fasta_file:
        fasta_file.write(f">{job_name}\n{rna_sequence}")

    preprocess_data(fasta_filepath, job_dir)
    run_nufold(fasta_filepath, job_base_dir, output_dir)

    # Zip the output directory
    shutil.make_archive(output_dir, 'zip', output_dir)
    print(f"NuFold output saved in {output_dir}.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NuFold locally")
    parser.add_argument("rna_sequence", type=str, help="RNA sequence")
    args = parser.parse_args()

    main(args.rna_sequence.upper())
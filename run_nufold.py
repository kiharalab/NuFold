import argparse
import os
import random
import sys
import pickle

import numpy as np
import torch
from nufold.config import model_config
from nufold.model.nufold import Nufold
from nufold.data import feature_pipeline, data_pipeline
from nufold.model.openfold.tensor_utils import tensor_tree_map
from nufold.common.nucleicacid import from_prediction, to_pdb
from nufold.common import base_constants

torch.set_grad_enabled(False)


def _main():
    """
    """
    args = argparser()

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    ITER = args.repeat

    config = model_config(args.config_preset)
    config.data.common.max_recycling_iters = args.recycle
    model = Nufold(config)
    model = model.eval()

    ckpt_path = args.ckpt_path
    d = torch.load(ckpt_path)
    d = d["ema"]["params"]

    model.load_state_dict(d)
    model = model.to("cuda:0")

    dp = data_pipeline.DataPipeline(
        template_featurizer=None,
        ss_enabled = True
    )

    random_seed = random.randrange(2**32)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    input_seqs = {}
    _n = ""
    _s = ""
    with open(args.input_fasta) as f:
        for line in f:
            if line.startswith(">"):
                if _n:
                    input_seqs[_n] = _s
                _n = line.strip()[1:]
                _s = ""
            else:
                _s += line.strip()
        if _n:
            input_seqs[_n] = _s

    for n, (d, s) in enumerate(input_seqs.items()):
        for i in range(ITER):
            print(f"{n}th: {d} is processing...")
            if os.path.exists(f"{OUTPUT_DIR}/{d}/{d}_unrelaxed{i}.pdb"):
                print(f"{OUTPUT_DIR}/{d}/{d}_unrelaxed{i}.pdb is exists, skipping...", file=sys.stderr)
                continue
            os.makedirs(f"{OUTPUT_DIR}/{d}", exist_ok=True)

            # read alignment
            try:
                feature_dict = generate_feature_dict(
                    d,
                    s,
                    INPUT_DIR,
                    dp,
                    f"{OUTPUT_DIR}/{d}",
                )
            except:
                print("parse error...")
                continue

            processed_feature_dict = feature_processor.process_features(
                feature_dict, mode='predict',
            )
            batch = {
                k:torch.as_tensor(v, device="cuda:0").unsqueeze(0)  # add batch dim
                for k,v in processed_feature_dict.items()
            }

            with torch.no_grad():
                out = model(batch)
                for j, _d in enumerate(out["recycling"]):
                    _d.update(model.aux_heads(_d))
                    _dd = tensor_tree_map(lambda x: np.array(x.cpu()), _d)
                    strcture = from_prediction(
                        tensor_tree_map(lambda x: np.array(x[..., j].cpu()), batch),
                        _dd,
                        b_factors = np.repeat(
                            _dd["plddt"][..., None], base_constants.atom_type_num, axis=-1
                        )
                    )
                    with open(f"{OUTPUT_DIR}/{d}/{d}_unrelaxed_rec_{j}.pdb", "wt") as f:
                        print(to_pdb(strcture), file=f)

                batch = tensor_tree_map(
                    lambda x: np.array(x[..., -1].cpu()),
                    batch
                )
                out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

            strcture = from_prediction(
                batch,
                out,
                b_factors = np.repeat(
                    out["plddt"][..., None], base_constants.atom_type_num, axis=-1
                )
            )
            with open(f"{OUTPUT_DIR}/{d}/{d}_unrelaxed{i}.pdb", "wt") as f:
                print(to_pdb(strcture), file=f)
            with open(f"{OUTPUT_DIR}/{d}/{d}_in_{i}.pkl", "wb") as ofh:
                pickle.dump(batch, ofh, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f"{OUTPUT_DIR}/{d}/{d}_out_{i}.pkl", "wb") as ofh:
                pickle.dump(out, ofh, protocol=pickle.HIGHEST_PROTOCOL)

    return


def generate_feature_dict(
    tag: str,
    seq: str,
    alignment_dir,
    data_processor,
    output_dir,
):
    tmp_fasta_path = os.path.join(output_dir, f"tmp_{os.getpid()}.fasta")

    with open(tmp_fasta_path, "w") as fp:
        fp.write(f">{tag}\n{seq}")

    local_alignment_dir = os.path.join(alignment_dir, tag)
    feature_dict = data_processor.process_fasta(
        fasta_path=tmp_fasta_path, alignment_dir=local_alignment_dir
    )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_preset", type=str, default="initial_training",
        help=(
            'Config setting. Choose e.g. "initial_training", "finetuning", '
            '"model_1", etc. By default, the actual values in the config are '
            'used.'
        )
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="",
    )
    parser.add_argument(
        "--input_fasta", type=str, default="",
    )
    parser.add_argument(
        "--input_dir", type=str, default="",
    )
    parser.add_argument(
        "--output_dir", type=str, default="",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
    )
    parser.add_argument(
        "--recycle", type=int, default=3,
    )


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    _main()

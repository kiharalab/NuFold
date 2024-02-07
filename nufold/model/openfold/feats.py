# Copyright 2022 Y.K, Kihara Lab
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
from typing import Dict

import nufold.common.base_constants as rc
from nufold.model.openfold.rigid_utils import Rotation, Rigid
from nufold.model.openfold.tensor_utils import (
    batched_gather,
    one_hot,
    tree_map,
    tensor_tree_map,
)


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    is_prn = (aatype == rc.restype_order["A"]) + (aatype == rc.restype_order["G"])
    n9_idx = rc.atom_order["N9"]
    n1_idx = rc.atom_order["N1"]

    pseudo_beta = torch.where(
        is_prn[..., None].expand(*((-1,) * len(is_prn.shape)), 3),
        all_atom_positions[..., n9_idx, :],
        all_atom_positions[..., n1_idx, :],
    )

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_prn,
            all_atom_masks[..., n9_idx],
            all_atom_masks[..., n1_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


def atom23_to_atom28(atom23, batch):
    atom28_data = batched_gather(
        atom23,
        batch["residx_atom28_to_atom23"],
        dim=-2,
        no_batch_dims=len(atom23.shape[:-2]),
    )

    atom28_data = atom28_data * batch["atom28_atom_exists"][..., None]

    return atom28_data


def build_ss_pair_feat(batch):
    aatype_one_hot = nn.functional.one_hot(
        batch["aatype"],
        rc.restype_num + 2,
    )
    n_res = batch["aatype"].shape[-1]

    # [*, N, N, 6]
    a = aatype_one_hot[..., None, :, :].expand(
        *aatype_one_hot.shape[:-2], n_res, -1, -1
    ).unsqueeze(1)
    # [*, N, N, 6]
    b = aatype_one_hot[..., None, :].expand(
        *aatype_one_hot.shape[:-2], -1, n_res, -1
    ).unsqueeze(1)
    ss_map = batch["ss_map"]
    return torch.cat([a, b, ss_map], dim=-1)


def build_template_angle_feat(template_feats):
    template_aatype = template_feats["template_aatype"]
    torsion_angles_sin_cos = template_feats["template_torsion_angles_sin_cos"]
    alt_torsion_angles_sin_cos = template_feats[
        "template_alt_torsion_angles_sin_cos"
    ]
    torsion_angles_mask = template_feats["template_torsion_angles_mask"]
    template_angle_feat = torch.cat(
        [
            nn.functional.one_hot(template_aatype, 6),  # was 22
            torsion_angles_sin_cos.reshape(
                *torsion_angles_sin_cos.shape[:-2], 7*2
            ),
            alt_torsion_angles_sin_cos.reshape(
                *alt_torsion_angles_sin_cos.shape[:-2], 7*2
            ),
            torsion_angles_mask,
        ],
        dim=-1,
    )

    return template_angle_feat


def build_template_pair_feat(
    batch, min_bin, max_bin, no_bins, eps=1e-20, inf=1e8
):
    template_mask = batch["template_pseudo_beta_mask"]
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

    # Compute distogram (this seems to differ slightly from Alg. 5)
    tpb = batch["template_pseudo_beta"]
    dgram = torch.sum(
        (tpb[..., None, :] - tpb[..., None, :, :]) ** 2, dim=-1, keepdim=True
    )
    lower = torch.linspace(min_bin, max_bin, no_bins, device=tpb.device) ** 2
    upper = torch.cat([lower[:-1], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)

    to_concat = [dgram, template_mask_2d[..., None]]

    aatype_one_hot = nn.functional.one_hot(
        batch["template_aatype"],
        rc.restype_num + 2,
    )

    n_res = batch["template_aatype"].shape[-1]
    to_concat.append(
        aatype_one_hot[..., None, :, :].expand(
            *aatype_one_hot.shape[:-2], n_res, -1, -1
        )
    )
    to_concat.append(
        aatype_one_hot[..., None, :].expand(
            *aatype_one_hot.shape[:-2], -1, n_res, -1
        )
    )

    # bad naming.
    n, ca, c = [rc.atom_order[a] for a in ["C4'", "C1'", "C3'"]]
    rigids = Rigid.make_transform_from_reference(
        n_xyz=batch["template_all_atom_positions"][..., n, :],
        ca_xyz=batch["template_all_atom_positions"][..., ca, :],
        c_xyz=batch["template_all_atom_positions"][..., c, :],
        eps=eps,
    )
    points = rigids.get_trans()[..., None, :, :]
    rigid_vec = rigids[..., None].invert_apply(points)

    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec ** 2, dim=-1))

    t_aa_masks = batch["template_all_atom_mask"]
    template_mask = (
        t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
    )
    template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

    inv_distance_scalar = inv_distance_scalar * template_mask_2d
    unit_vector = rigid_vec * inv_distance_scalar[..., None]
    to_concat.extend(torch.unbind(unit_vector[..., None, :], dim=-1))
    to_concat.append(template_mask_2d[..., None])

    act = torch.cat(to_concat, dim=-1)
    act = act * template_mask_2d[..., None]

    return act


def build_extra_msa_feat(batch):
    msa_1hot = nn.functional.one_hot(batch["extra_msa"], 7)  # was 23
    msa_feat = [
        msa_1hot,
        batch["extra_has_deletion"].unsqueeze(-1),
        batch["extra_deletion_value"].unsqueeze(-1),
    ]
    return torch.cat(msa_feat, dim=-1)


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
):
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    g1_frame_to_bb = all_frames[..., 1]
    g2_frame_to_frame = all_frames[..., 2]
    g2_frame_to_bb = g1_frame_to_bb.compose(g2_frame_to_frame)
    g3_frame_to_frame = all_frames[..., 3]
    g3_frame_to_bb = g2_frame_to_bb.compose(g3_frame_to_frame)
    g4_frame_to_frame = all_frames[..., 4]
    g4_frame_to_bb = g3_frame_to_bb.compose(g4_frame_to_frame)
    g5_frame_to_frame = all_frames[..., 5]
    g5_frame_to_bb = g4_frame_to_bb.compose(g5_frame_to_frame)
    g6_frame_to_bb = all_frames[..., 6]
    g7_frame_to_bb = all_frames[..., 7]
    g8_frame_to_frame = all_frames[..., 8]
    g8_frame_to_bb = g7_frame_to_bb.compose(g8_frame_to_frame)
    g9_frame_to_bb = all_frames[..., 9]

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :1],
            g1_frame_to_bb.unsqueeze(-1),
            g2_frame_to_bb.unsqueeze(-1),
            g3_frame_to_bb.unsqueeze(-1),
            g4_frame_to_bb.unsqueeze(-1),
            g5_frame_to_bb.unsqueeze(-1),
            g6_frame_to_bb.unsqueeze(-1),
            g7_frame_to_bb.unsqueeze(-1),
            g8_frame_to_bb.unsqueeze(-1),
            g9_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom23_pos(  # was 14
    r: Rigid,
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 23, 4, 4]
    default_4x4 = default_frames[aatype, ...]

    # [*, N, 23]
    group_mask = group_idx[aatype, ...]

    # [*, N, 23, 8]
    group_mask = nn.functional.one_hot(
        group_mask.long(),  # somehow
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 23, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 23]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 23, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 23, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions
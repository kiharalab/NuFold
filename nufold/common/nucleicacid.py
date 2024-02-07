#!/usr/bin/env python3

import dataclasses
import io
from typing import Any, Mapping, Optional

from Bio.PDB import PDBParser
import numpy as np

from nufold.common import base_constants

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class NucleicAcid:
    """Nucleic acid structure representation"""
    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are ...? .
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Nucleotide type for each base represented as an integer between 0 and 4,
    # where 4 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean value.
    b_factors: np.ndarray  # [num_res, num_atom_type]


def from_pdb_string(
    pdb_str: str,
    chain_id: Optional[str] = None
) -> NucleicAcid:
    """Takes a PDB string and constructs a NucleicAcid object.

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Args:
        pdb_str: The contents of the pdb file
        chain_id: If None, then the pdb file must contain a single chain (which
        will be parsed). If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
        A new `NucleicAcid` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f'Only single model PDBs are supported. Found {len(models)} models.'
        )
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != ' ':
                raise ValueError(
                    f'PDB contains an insertion code at chain {chain.id} and residue '
                    f'index {res.id[1]}. These are not supported.'
                )
            res_shortname = base_constants.restype_3to1.get(res.resname, 'X')
            restype_idx = base_constants.restype_order.get(
                res_shortname, base_constants.restype_num
            )
            pos = np.zeros((base_constants.atom_type_num, 3))
            mask = np.zeros((base_constants.atom_type_num,))
            res_b_factors = np.zeros((base_constants.atom_type_num,))
            for atom in res:
                if atom.name not in base_constants.atom_types:
                    continue
                pos[base_constants.atom_order[atom.name]] = atom.coord
                mask[base_constants.atom_order[atom.name]] = 1.
                res_b_factors[base_constants.atom_order[atom.name]] = atom.bfactor

            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)
    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return NucleicAcid(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        b_factors=np.array(b_factors)
    )


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = 'TER'
    return (
        f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
        f'{chain_name:>1}{residue_index:>4}'
    )


def to_pdb(nuclacid: NucleicAcid) -> str:
    """Converts a `NucleicAcid` instance to a PDB string.
    Args:
      nuclacid: The protein to convert to PDB.
    Returns:
      PDB string.
    """
    restypes = base_constants.restypes + ["X"]
    res_1to3 = lambda r: base_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = base_constants.atom_types

    pdb_lines = []

    atom_mask = nuclacid.atom_mask
    aatype = nuclacid.aatype
    atom_positions = nuclacid.atom_positions
    residue_index = nuclacid.residue_index.astype(np.int32)
    b_factors = nuclacid.b_factors

    if np.any(aatype > base_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    pdb_lines.append("MODEL     1")
    atom_index = 1
    chain_id = "A"
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[
                0
            ]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_id:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the chain.
    chain_end = "TER"
    chain_termination_line = (
        f"{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[-1]):>3} "
        f"{chain_id:>1}{residue_index[-1]:>4}"
    )
    pdb_lines.append(chain_termination_line)
    pdb_lines.append("ENDMDL")

    pdb_lines.append("END")
    pdb_lines.append("")
    return "\n".join(pdb_lines)


def ideal_atom_mask(nuclacid: NucleicAcid) -> np.ndarray:
    """Computes an ideal atom mask.
    `NucleicAcid.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.
    Args:
        prot: `NucleicAcid` whose fields are `numpy.ndarray` objects.
    Returns:
        An ideal atom mask.
    """
    return base_constants.STANDARD_ATOM_MASK[nuclacid.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = True,
) -> NucleicAcid:
    """Assembles a protein from a prediction.
    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
    Returns:
      A protein instance.
    """
    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr

    if b_factors is None:
        b_factors = np.zeros_like(result["final_atom_mask"])

    return NucleicAcid(
        aatype=_maybe_remove_leading_dim(features["aatype"]),
        atom_positions=_maybe_remove_leading_dim(result["final_atom_positions"]),
        atom_mask=_maybe_remove_leading_dim(result["final_atom_mask"]),
        residue_index=_maybe_remove_leading_dim(features["residue_index"]) + 1,
        b_factors=_maybe_remove_leading_dim(b_factors),
    )

"""Constraints used in NuFold."""

import collections
from typing import Mapping, List, Tuple
import functools
import os
import numpy as np
import tree

# Distance from one P to next P
# This value was taken from observation, but it has more diversity.
p_p = 6.137854156947696  # std == 1.2751798156277778

# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
chi_angles_atoms = {
    'A': [
        ["O4'", "C1'", "N9", "C8"],  # chi1
    ],
    'G': [
        ["O4'", "C1'", "N9", "C8"],  # chi1
    ],
    'C': [
        ["O4'", "C1'", "N1", "C2"],  # chi1
    ],
    'U': [
        ["O4'", "C1'", "N1", "C2"],   # chi1
    ],
}
#
# Those definitions are temporary. The concept of those chi angles are like
# once chi2 is not determined if chi1 was not determined, but I broke that sense.
#

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
chi_angles_mask = [
    [1.0],  # ADE
    [1.0],  # GUA
    [1.0],  # CYT
    [1.0],  # URA
]

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
chi_pi_periodic = [
    [0.0],  # ADE
    [0.0],  # GUA
    [0.0],  # CYT
    [0.0],  # URA
    [0.0],  # UNK
]

# Atoms positions relative to the 8 rigid groups, defined by the pre-omega, phi,
# psi and chi angles:
# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_idx, rel_position]
# Atoms positions relative to the XXX rigid groups, defined by X, X, X, and X
# 0: 'sugar ring group'
# 1: 'backbone post-epsilon (empty)'
# 2: 'backbone pre-zeta (empty)'
# 3: 'backbone r_delta (included in 0th frame)'
# 4: 'backbone r_ganma'
# 5: 'backbone r_beta'
# 6: 'backbone r_alpha'
# 7: 'base group (chi)'
rigid_group_atom_positions = {
    'A': [
        ["P", 4, (0.793, 1.382, -0.000),],
        ["OP1", 5, (0.428, 1.418, 0.000),],
        ["OP2", 5, (0.504, -0.917, 1.039),],
        ["O5'", 3, (0.499, 1.332, 0.000),],
        ["C5'", 2, (0.511, 1.417, 0.000),],
        ["C4'", 1, (0.484, 1.366, -0.000),],
        ["O4'", 0, (-0.421, 1.352, 0.000),],
        ["C3'", 7, (0.291, 1.494, -0.000),],
        ["O3'", 8, (0.534, 1.325, 0.000),],
        ["C2'", 0, (1.527, -0.000, 0.000),],
        ["O2'", 6, (0.493, 1.323, 0.000),],
        ["C1'", 0, (0.000, 0.000, 0.000),],
        ["N9", 0, (-0.567, -0.656, -1.192)],
        ["C8", 9 , (0.842, 1.083, 0.000)],
        ["N7", 9 , (2.112, 0.765, -0.008)],
        ["C5", 9 , (2.105, -0.622, -0.018)],
        ["C6", 9 , (3.138, -1.571, -0.030)],
        ["N6", 9 , (4.432, -1.249, -0.035)],
        ["N1", 9 , (2.794, -2.877, -0.037)],
        ["C2", 9 , (1.495, -3.198, -0.032)],
        ["N3", 9 , (0.433, -2.397, -0.023)],
        ["C4", 9 , (0.812, -1.107, -0.015)],
    ],
    'G': [
        ["P", 4, (0.793, 1.382, -0.000),],
        ["OP1", 5, (0.428, 1.418, 0.000),],
        ["OP2", 5, (0.504, -0.917, 1.039),],
        ["O5'", 3, (0.499, 1.332, 0.000),],
        ["C5'", 2, (0.511, 1.417, 0.000),],
        ["C4'", 1, (0.484, 1.366, -0.000),],
        ["O4'", 0, (-0.421, 1.352, 0.000),],
        ["C3'", 7, (0.291, 1.494, -0.000),],
        ["O3'", 8, (0.534, 1.325, 0.000),],
        ["C2'", 0, (1.527, -0.000, 0.000),],
        ["O2'", 6, (0.493, 1.323, 0.000),],
        ["C1'", 0, (0.000, 0.000, 0.000),],
        ["N9", 0, (-0.567, -0.656, -1.192)],
        ["C8",  9, (0.834, 1.092, -0.000)],
        ["N7",  9, (2.099, 0.771, -0.003)],
        ["C5",  9, (2.100, -0.616, -0.009)],
        ["C6",  9, (3.181, -1.532, -0.013)],
        ["O6",  9, (4.394, -1.288, -0.015)],
        ["N1",  9, (2.737, -2.849, -0.018)],
        ["C2",  9, (1.423, -3.238, -0.017)],
        ["N2",  9, (1.197, -4.559, -0.022)],
        ["N3",  9, (0.405, -2.394, -0.012)],
        ["C4", 9, (0.814, -1.108, -0.008)],
    ],
    'C': [
        ["P", 4, (0.793, 1.382, -0.000),],
        ["OP1", 5, (0.428, 1.418, 0.000),],
        ["OP2", 5, (0.504, -0.917, 1.039),],
        ["O5'", 3, (0.499, 1.332, 0.000),],
        ["C5'", 2, (0.511, 1.417, 0.000),],
        ["C4'", 1, (0.484, 1.366, -0.000),],
        ["O4'", 0, (-0.421, 1.352, 0.000),],
        ["C3'", 7, (0.291, 1.494, -0.000),],
        ["O3'", 8, (0.534, 1.325, 0.000),],
        ["C2'", 0, (1.527, -0.000, 0.000),],
        ["O2'", 6, (0.493, 1.323, 0.000),],
        ["C1'", 0, (0.000, 0.000, 0.000),],
        ["N1", 0, (-0.567, -0.656, -1.192),],
        ["C2", 9, (0.664, 1.227, 0.000),],
        ["O2", 9, (-0.012, 2.271, -0.001),],
        ["N3", 9, (2.020, 1.245, 0.004),],
        ["C4", 9, (2.702, 0.093, 0.012),],
        ["N4", 9, (4.043, 0.152, 0.024),],
        ["C5", 9, (2.045, -1.171, 0.011),],
        ["C6", 9, (0.710, -1.172, 0.004),],
    ],
    'U': [
        ["P", 4, (0.793, 1.382, -0.000),],
        ["OP1", 5, (0.428, 1.418, 0.000),],
        ["OP2", 5, (0.504, -0.917, 1.039),],
        ["O5'", 3, (0.499, 1.332, 0.000),],
        ["C5'", 2, (0.511, 1.417, 0.000),],
        ["C4'", 1, (0.484, 1.366, -0.000),],
        ["O4'", 0, (-0.421, 1.352, 0.000),],
        ["C3'", 7, (0.291, 1.494, -0.000),],
        ["O3'", 8, (0.534, 1.325, 0.000),],
        ["C2'", 0, (1.527, -0.000, 0.000),],
        ["O2'", 6, (0.493, 1.323, 0.000),],
        ["C1'", 0, (0.000, 0.000, 0.000),],
        ["N1", 0, (-0.567, -0.656, -1.192)],
        ["C2",  9, (0.634, 1.228, -0.000)],
        ["O2",  9, (0.028, 2.285, -0.007)],
        ["N3",  9, (2.005, 1.175, 0.004)],
        ["C4",  9, (2.793, 0.042, 0.012)],
        ["O4",  9, (4.020, 0.158, 0.014)],
        ["C5",  9, (2.063, -1.190, 0.013)],
        ["C6",  9, (0.726, -1.169, 0.010)],
    ],
}
orig_rigid_group_atom_positions = {
    # NOTE: chainID is from Auth, but resi is from Label.
    # 'A': "6YDP_BB_23A",
    # 'G': "6PMO_B_65C",
    # 'C': "7O80_AT_63G",
    # 'U': "5B2T_A_80U",
}

# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
_base_atoms = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
residue_atoms = {
    'A': _base_atoms + ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    'G': _base_atoms + ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    'C': _base_atoms + ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    'U': _base_atoms + ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
}

# Naming swaps for ambiguous atom names.
residue_atom_renaming_swaps = {
    'A': {'OP1': 'OP2'},
    'G': {'OP1': 'OP2'},
    'C': {'OP1': 'OP2'},
    'U': {'OP1': 'OP2'}
}

# Van der Waals radii [Angstroem] of the atoms (from Wikipedia)
# https://en.wikipedia.org/wiki/Van_der_Waals_radius
van_der_waals_radius = {
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'S': 1.8,
    'P': 1.8,
}

Bond = collections.namedtuple(
    'Bond', ['atom1_name', 'atom2_name', 'length', 'stddev']
)
BondAngle = collections.namedtuple(
    'BondAngle',
    ['atom1_name', 'atom2_name', 'atom3name', 'angle_rad', 'stddev']
)


@functools.lru_cache(maxsize=None)
def load_stereo_chemical_props() -> Tuple[Mapping[str, List[Bond]],
                                          Mapping[str, List[Bond]],
                                          Mapping[str, List[BondAngle]]]:
    # parameters are taken from (Parkinson etal 1996) and (Gilski etal 2019)
    stereo_chemical_props_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'stereo_chemical_props_rna.txt'
    )
    with open(stereo_chemical_props_path, 'rt') as f:
        stereo_chemical_props = f.read()
    lines_iter = iter(stereo_chemical_props.splitlines())
    # Load bond lengths.
    residue_bonds = {}
    next(lines_iter)  # Skip header line.
    for line in lines_iter:
        if line.strip() == '-':
            break
        bond, resname, length, stddev = line.split()
        atom1, atom2 = bond.split('-')
        if resname not in residue_bonds:
            residue_bonds[resname] = []
        residue_bonds[resname].append(
            Bond(atom1, atom2, float(length), float(stddev))
        )
    residue_bonds['UNK'] = []

    # Load bond angles.
    residue_bond_angles = {}
    next(lines_iter)  # Skip empty line.
    next(lines_iter)  # Skip header line.
    for line in lines_iter:
        if line.strip() == '-':
            break
        bond, resname, angle_degree, stddev_degree = line.split()
        atom1, atom2, atom3 = bond.split('-')
        if resname not in residue_bond_angles:
            residue_bond_angles[resname] = []
        residue_bond_angles[resname].append(
            BondAngle(
                atom1,
                atom2,
                atom3,
                float(angle_degree) / 180. * np.pi,
                float(stddev_degree) / 180. * np.pi
            )
        )
    residue_bond_angles['UNK'] = []

    def make_bond_key(atom1_name, atom2_name):
        """Unique key to lookup bonds."""
        return '-'.join(sorted([atom1_name, atom2_name]))

    # Translate bond angles into distances ("virtual bonds").
    residue_virtual_bonds = {}
    for resname, bond_angles in residue_bond_angles.items():
        # Create a fast lookup dict for bond lengths.
        bond_cache = {}
        for b in residue_bonds[resname]:
            bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b
        residue_virtual_bonds[resname] = []
        for ba in bond_angles:
            bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
            bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]
            # Compute distance between atom1 and atom3 using the law of cosines
            # c^2 = a^2 + b^2 - 2ab*cos(gamma).
            gamma = ba.angle_rad
            length = np.sqrt(
                bond1.length**2 + bond2.length**2
                - 2 * bond1.length * bond2.length * np.cos(gamma)
            )

            # Propagation of uncertainty assuming uncorrelated errors.
            dl_outer = 0.5 / length
            dl_dgamma = (2 * bond1.length * bond2.length * np.sin(gamma)) * dl_outer
            dl_db1 = (2 * bond1.length - 2 * bond2.length * np.cos(gamma)) * dl_outer
            dl_db2 = (2 * bond2.length - 2 * bond1.length * np.cos(gamma)) * dl_outer
            stddev = np.sqrt(
                (dl_dgamma * ba.stddev)**2 +
                (dl_db1 * bond1.stddev)**2 +
                (dl_db2 * bond2.stddev)**2
            )
            residue_virtual_bonds[resname].append(
                Bond(ba.atom1_name, ba.atom3name, length, stddev)
            )
    return (residue_bonds, residue_virtual_bonds, residue_bond_angles)


# Between-residue bond lengths for general bonds
between_res_bond_length_o3_p = [1.607]             # between_res_bond_length_c_n
between_res_bond_length_stddev_o3_p = [0.012]      # between_res_bond_length_stddev_c_n
# Between-residue cos_angles.
between_res_cos_angles_o3_p_o5 = [-0.2433, 0.0052]  # degrees: 104.083 +- 5.861
between_res_cos_angles_c3_o3_p = [-0.4984, 0.0031]  # degrees: 119.893 +- 4.493

atom_types = [  # L=28
    "P",   # 0
    "OP1", # 1
    "OP2", # 2
    "O5'", # 3
    "C5'", # 4
    "C4'", # 5
    "O4'", # 6
    "C3'",  # 7
    "O3'",  # 8
    "C2'", # 9
    "O2'", # 10
    "C1'", # 11
    "N9", # 12
    "C8", #
    "N7", #
    "C5",
    "C6",
    "N6",
    "N1", # 18
    "C2",
    "N3",
    "C4",
    "O6",
    "N2",
    "O2",
    "N4",
    "O4",
    "OP3",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 28

# A compact atom encoding with 23 columns
restype_name_to_atom23_names = {
    'A': list(residue_atoms["A"]) + [''],
    'G': list(residue_atoms["G"]) + [],
    'C': list(residue_atoms["C"]) + ['', '', ''],
    'U': list(residue_atoms["U"]) + ['', '', ''],
}

# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
restypes = ['A', 'G', 'C', 'U']
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_order["T"] = 3
restype_num = len(restypes)  # := 4.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.
restypes_with_x = restypes + ['X']
restypes_with_x_g = restypes_with_x + ["-"]
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}
restype_order_with_x["T"] = 3
restype_order_with_x_and_gap = {restype: i for i, restype in enumerate(restypes_with_x_g)}
restype_order_with_x_and_gap["T"] = 3
for r in "WSMKRYBDHVN":  # map ambiguous bases into unknown
    restype_order_with_x_and_gap[r] = unk_restype_index
rev_restype_order_with_x_and_gap = {
    0: "A",
    1: "G",
    2: "C",
    3: "U",
    4: "X",
    5: "-",
}
map_id_to_out_aatype = tuple(
    restypes_with_x_g.index(rev_restype_order_with_x_and_gap[i])
    for i in range(len(restypes_with_x_g))
)

def sequence_to_onehot(
    sequence: str,
    mapping: Mapping[str, int],
    map_unknown_to_x: bool = False) -> np.ndarray:
    """Maps the given sequence into a one-hot encoded matrix."""

    num_entries = max(mapping.values()) + 1
    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError(
            'The mapping must have values from 0 to num_unique_aas-1 '
            'without any gaps. Got: %s' % sorted(mapping.values())
        )
    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)
    for aa_index, aa_type in enumerate(sequence):
        if map_unknown_to_x:
            if aa_type.isalpha() and aa_type.isupper():
                aa_id = mapping.get(aa_type, mapping['X'])
            else:
                raise ValueError(f'Invalid character in the sequence: {aa_type}')
        else:
            aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1
    return one_hot_arr

restype_1to3 = {
    'A': 'A',
    'G': 'G',
    'C': 'C',
    'U': 'U',
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Define a restype name for all unknown residues.
unk_restype = 'UNK'
resnames = [restype_1to3[r] for r in restypes] + [unk_restype]
resname_to_idx = {resname: i for i, resname in enumerate(resnames)}

# COMMENT: HHBLITS IS NOT USED HERE FOR RNA

def _make_standard_atom_mask() -> np.ndarray:
  """Returns [num_res_types, num_atom_types] mask array."""
  # +1 to account for unknown (all 0s).
  mask = np.zeros([restype_num + 1, atom_type_num], dtype=np.int32)
  for restype, restype_letter in enumerate(restypes):
    restype_name = restype_1to3[restype_letter]
    atom_names = residue_atoms[restype_name]
    for atom_name in atom_names:
      atom_type = atom_order[atom_name]
      mask[restype, atom_type] = 1
  return mask

STANDARD_ATOM_MASK = _make_standard_atom_mask()


# A one hot representation for the first and second atoms defining the axis
# of rotation for each chi-angle in each residue.
def chi_angle_atom(atom_index: int) -> np.ndarray:
    """Define chi-angle rigid groups via one-hot representations."""
    chi_angles_index = {}
    one_hots = []

    for k, v in chi_angles_atoms.items():
        indices = [atom_types.index(s[atom_index]) for s in v]
        indices.extend([-1]*(4-len(indices)))
        chi_angles_index[k] = indices
    for r in restypes:
        res3 = restype_1to3[r]
        one_hot = np.eye(atom_type_num)[chi_angles_index[res3]]
        one_hots.append(one_hot)
    one_hots.append(np.zeros([4, atom_type_num]))  # Add zeros for residue `X`.
    one_hot = np.stack(one_hots, axis=0)
    one_hot = np.transpose(one_hot, [0, 2, 1])

    return one_hot

chi_atom_1_one_hot = chi_angle_atom(1)
chi_atom_2_one_hot = chi_angle_atom(2)

# An array like chi_angles_atoms but using indices rather than names.
chi_angles_atom_indices = [chi_angles_atoms[restype_1to3[r]] for r in restypes]
chi_angles_atom_indices = tree.map_structure(
    lambda atom_name: atom_order[atom_name], chi_angles_atom_indices
)
chi_angles_atom_indices = np.array([
    chi_atoms + ([[0, 0, 0, 0]] * (4 - len(chi_atoms)))
    for chi_atoms in chi_angles_atom_indices
])

# Mapping from (res_name, atom_name) pairs to the atom's chi group index
# and atom index within that group.
chi_groups_for_atom = collections.defaultdict(list)
for res_name, chi_angle_atoms_for_res in chi_angles_atoms.items():
    for chi_group_i, chi_group in enumerate(chi_angle_atoms_for_res):
        for atom_i, atom in enumerate(chi_group):
            chi_groups_for_atom[(res_name, atom)].append((chi_group_i, atom_i))
chi_groups_for_atom = dict(chi_groups_for_atom)


def _make_rigid_transformation_4x4(ex, ey, translation):
    """Create a rigid 4x4 transformation matrix from two axes and transl."""
    # Normalize ex.
    ex_normalized = ex / np.linalg.norm(ex)

    # make ey perpendicular to ex
    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)

    # compute ez as cross product
    eznorm = np.cross(ex_normalized, ey_normalized)
    m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
    m = np.concatenate([m, [[0., 0., 0., 1.]]], axis=0)
    return m

# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
# and compute affine transformation matrices (4,4) from one rigid group to the
# previous group
restype_atom28_to_rigid_group = np.zeros([5, 28], dtype=np.int32)
restype_atom28_mask = np.zeros([5, 28], dtype=np.float32)
restype_atom28_rigid_group_positions = np.zeros([5, 28, 3], dtype=np.float32)
restype_atom23_to_rigid_group = np.zeros([5, 23], dtype=np.int32)
restype_atom23_mask = np.zeros([5, 23], dtype=np.float32)
restype_atom23_rigid_group_positions = np.zeros([5, 23, 3], dtype=np.float32)
restype_rigid_group_default_frame = np.zeros([5, 10, 4, 4], dtype=np.float32)


def _make_rigid_group_constants():
    """Fill the arrays above."""
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        for atomname, group_idx, atom_position in rigid_group_atom_positions[resname]:
            atomtype = atom_order[atomname]
            restype_atom28_to_rigid_group[restype, atomtype] = group_idx
            restype_atom28_mask[restype, atomtype] = 1
            restype_atom28_rigid_group_positions[restype, atomtype, :] = atom_position

            atom23idx = restype_name_to_atom23_names[resname].index(atomname)
            restype_atom23_to_rigid_group[restype, atom23idx] = group_idx
            restype_atom23_mask[restype, atom23idx] = 1
            restype_atom23_rigid_group_positions[restype, atom23idx, :] = atom_position

    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        atom_positions = {
            name: np.array(pos) for name, _, pos in rigid_group_atom_positions[resname]
        }
        if resname in ["A", "G"]:
            first_N = "N9"
        elif resname in ["U", "C"]:
            first_N = "N1"
        # backbone to backbone is the identity transform
        restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)

        # group 1: add C4'
        # C2' -> C1' -> O4' -> C4'
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["O4'"] - atom_positions["C1'"],
            ey=atom_positions["C2'"] - atom_positions["C1'"],
            translation=atom_positions["O4'"])
        restype_rigid_group_default_frame[restype, 1, :, :] = mat

        # group 2: add C5' after frame 1
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C4'"],
            ey=np.array([-1.0, 0.0, 0.0]),
            translation=atom_positions["C4'"]
        )
        restype_rigid_group_default_frame[restype, 2, :, :] = mat

        # group 3: add O5' after frame 2
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C5'"],
            ey=np.array([-1.0, 0.0, 0.0]),
            translation=atom_positions["C5'"]
        )
        restype_rigid_group_default_frame[restype, 3, :, :] = mat

        # group 4: add P after frame 3
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["O5'"],
            ey=np.array([-1.0, 0.0, 0.0]),
            translation=atom_positions["O5'"]
        )
        restype_rigid_group_default_frame[restype, 4, :, :] = mat

        # group 5: add OP1 after frame 4
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["P"],
            ey=np.array([-1.0, 0.0, 0.0]),
            translation=atom_positions["P"]
        )
        restype_rigid_group_default_frame[restype, 5, :, :] = mat

        # group 6: add O2'
        # O4' -> C1' -> C2' -> O2'
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C2'"] - atom_positions["C1'"],
            ey=atom_positions["O4'"] - atom_positions["C1'"],
            translation=atom_positions["C2'"])
        restype_rigid_group_default_frame[restype, 6, :, :] = mat

        # group 7: add C3'
        # O4' -> C1' -> C2' -> C3'
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C2'"] - atom_positions["C1'"],
            ey=atom_positions["O4'"] - atom_positions["C1'"],
            translation=atom_positions["C2'"])
        restype_rigid_group_default_frame[restype, 7, :, :] = mat

        # group 8: add O3' after frame 7
        mat = _make_rigid_transformation_4x4(
            ex=atom_positions["C3'"],
            ey=np.array([-1.0, 0.0, 0.0]),
            translation=atom_positions["C3'"]
        )
        restype_rigid_group_default_frame[restype, 8, :, :] = mat

        # group 9: add base
        if chi_angles_mask[restype][0]:
            base_atom_names = chi_angles_atoms[resname][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
                translation=base_atom_positions[2])
            restype_rigid_group_default_frame[restype, 9, :, :] = mat

_make_rigid_group_constants()


def make_atom23_dists_bounds(overlap_tolerance=1.5,
                             bond_length_tolerance_factor=15):
    """compute upper and lower bounds for bonds to assess violations."""
    restype_atom23_bond_lower_bound = np.zeros([5, 23, 23], np.float32)
    restype_atom23_bond_upper_bound = np.zeros([5, 23, 23], np.float32)
    restype_atom23_bond_stddev = np.zeros([5, 23, 23], np.float32)
    residue_bonds, residue_virtual_bonds, _ = load_stereo_chemical_props()
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        atom_list = restype_name_to_atom23_names[resname]

        # create lower and upper bounds for clashes
        for atom1_idx, atom1_name in enumerate(atom_list):
            if not atom1_name:
                continue
            atom1_radius = van_der_waals_radius[atom1_name[0]]
            for atom2_idx, atom2_name in enumerate(atom_list):
                if (not atom2_name) or atom1_idx == atom2_idx:
                    continue
                atom2_radius = van_der_waals_radius[atom2_name[0]]
                lower = atom1_radius + atom2_radius - overlap_tolerance
                upper = 1e10
                restype_atom23_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
                restype_atom23_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
                restype_atom23_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
                restype_atom23_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper

            # overwrite lower and upper bounds for bonds and angles
            for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
                atom1_idx = atom_list.index(b.atom1_name)
                atom2_idx = atom_list.index(b.atom2_name)
                lower = b.length - bond_length_tolerance_factor * b.stddev
                upper = b.length + bond_length_tolerance_factor * b.stddev
                restype_atom23_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
                restype_atom23_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
                restype_atom23_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
                restype_atom23_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
                restype_atom23_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
                restype_atom23_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
    return {
        'lower_bound': restype_atom23_bond_lower_bound,  # shape (5,23,23)
        'upper_bound': restype_atom23_bond_upper_bound,  # shape (5,23,23)
        'stddev': restype_atom23_bond_stddev,  # shape (5,23,23)
    }


if __name__ == "__main__":
    import torch
    default_frames = torch.tensor(restype_rigid_group_default_frame, requires_grad=False,)
    group_idx = torch.tensor(restype_atom23_to_rigid_group, requires_grad=False,)
    atom_mask = torch.tensor(restype_atom23_mask, requires_grad=False,)
    lit_positions = torch.tensor(restype_atom23_rigid_group_positions, requires_grad=False,)
    print(lit_positions)


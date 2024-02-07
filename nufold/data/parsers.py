#!/usr/bin/env python3
#
"""Functions for parsing various file formats."""

import collections
import dataclasses
import io
import string
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set
from Bio import AlignIO

DeletionMatrix = Sequence[Sequence[int]]


@dataclasses.dataclass(frozen=True)
class Msa:
    """Class representing a parsed MSA file."""
    sequences: Sequence[str]
    deletion_matrix: DeletionMatrix
    descriptions: Sequence[str]

    def __post_init__(self):
        if not (
            len(self.sequences) ==
            len(self.deletion_matrix) ==
            len(self.descriptions)
        ):
            raise ValueError(
                'All fields for an MSA must have the same length. '
                f'Got {len(self.sequences)} sequences, '
                f'{len(self.deletion_matrix)} rows in the deletion matrix and '
                f'{len(self.descriptions)} descriptions.'
            )

    def __len__(self):
        return len(self.sequences)

    def truncate(self, max_seqs: int):
        return Msa(
            sequences=self.sequences[:max_seqs],
            deletion_matrix=self.deletion_matrix[:max_seqs],
            descriptions=self.descriptions[:max_seqs]
        )


def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.
    Arguments:
        fasta_string: The string contents of a FASTA file.
    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append('')
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line
    return sequences, descriptions


def parse_stockholm(
    stockholm_string: str,
    query_sequence: Optional[str] = None
) -> Tuple[Sequence[str], DeletionMatrix, Sequence[str]]:
    """Parses sequences and deletion matrix from stockholm format alignment.
    Args:
        stockholm_string: The string contents of a stockholm file. The first
            sequence in the file should be the query sequence.
    Returns:
        A tuple of:
            * A list of sequences that have been aligned to the query. These
                might contain duplicates.
            * The deletion matrix for the alignment as a list of lists. The element
                at `deletion_matrix[i][j]` is the number of residues deleted from
                the aligned sequence i at residue position j.
            * The names of the targets matched, including the jackhmmer subsequence
                suffix.
    """
    name_to_sequence = collections.OrderedDict()

    sto_fh = io.StringIO(stockholm_string)
    align = AlignIO.read(sto_fh, "stockholm")
    for record in align:
        name = record.id
        seq = record.seq
        if name not in name_to_sequence:
            name_to_sequence[name] = ""
        name_to_sequence[name] += seq

    msa = []
    deletion_matrix = []

    query = query_sequence
    keep_columns = []
    for seq_index, sequence in enumerate(name_to_sequence.values()):
        if seq_index == 0:
            # Gather the columns with gaps from the query
            if not query:
                query = sequence
            keep_columns = [i for i, res in enumerate(query) if res != "-"]

        # Remove the columns with gaps in the query from all sequences.
        aligned_sequence = "".join([sequence[c] for c in keep_columns])

        msa.append(aligned_sequence)

        # Count the number of deletions w.r.t. query.
        deletion_vec = []
        deletion_count = 0
        for seq_res, query_res in zip(sequence, query):
            if seq_res != "-" or query_res != "-":
                if query_res == "-":
                    deletion_count += 1
                else:
                    deletion_vec.append(deletion_count)
                    deletion_count = 0
        deletion_matrix.append(deletion_vec)
    return msa, deletion_matrix, list(name_to_sequence.keys())


def parse_a3m(a3m_string: str) -> Tuple[Sequence[str], DeletionMatrix]:
    """Parses sequences and deletion matrix from a3m format alignment.
    Args:
        a3m_string: The string contents of a a3m file. The first sequence in the
            file should be the query sequence.
    Returns:
        A tuple of:
            * A list of sequences that have been aligned to the query. These
                might contain duplicates.
            * The deletion matrix for the alignment as a list of lists. The element
                at `deletion_matrix[i][j]` is the number of residues deleted from
                the aligned sequence i at residue position j.
    """
    sequences, _ = parse_fasta(a3m_string)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return aligned_sequences, deletion_matrix


def db2cmap(ss_string: str):
    res = []
    brackets = [("(", ")"), ("[", "]"), ("{", "}")]
    for ob, cb in brackets:
        tmp = []
        for i, s in enumerate(ss_string):
            if s == ob:
                tmp.append(i)
            elif s == cb:
                res.append((tmp.pop(), i))
    mat = [[[0] for j in range(len(ss_string))] for i in range(len(ss_string))]
    for i, j in res:
        mat[i][j] = [1]
        mat[j][i] = [1]
    return mat


def parse_dot_bracket(
    ss_string: str,
    format: Optional[str] = None
    ):
    """_summary_

    Args:
        ss_string (str): content of ss prediction
    """
    if format == ".ipknot":
        # 1st: header
        # 2nd: sequence
        # 3rd: dot-bracket
        db = ss_string.splitlines()[2]
        return db2cmap(db)
    else:
        return db2cmap(ss_string)
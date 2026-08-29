"""
esmfold_benchmark.py

Benchmarks ESMFold structure predictions against experimental PDB structures
using TM-score and Kabsch-aligned RMSD. Meant to run alongside your
Streamlit app (protein_analyzer.py) as the actual research/benchmarking
pipeline for IRIS.

Usage:
    pip install requests numpy --break-system-packages
    python esmfold_benchmark.py
"""
import streamlit.components.v1 as components
from gettext import install

import pip
import streamlit as st 

import csv
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import requests

MAX_ESMFOLD_LENGTH = 400  # ESM Atlas API practical limit

from Bio.PDB import PDBParser, PDBIO, Superimposer
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
import io
import re

CHUNK_OVERLAP = 50  # residues shared between adjacent chunks, used for alignment

def chunk_sequence(sequence: str, max_length: int = MAX_ESMFOLD_LENGTH, overlap: int = CHUNK_OVERLAP):
    """Split a long sequence into overlapping windows, each <= max_length."""
    chunks = []
    step = max_length - overlap
    start = 0
    while start < len(sequence):
        end = min(start + max_length, len(sequence))
        chunks.append((start, end, sequence[start:end]))
        if end == len(sequence):
            break
        start += step
    return chunks


def _parse_pdb_text(pdb_text: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("chunk", io.StringIO(pdb_text))
    return structure[0]["A"]  # ESMFold single-chain output, chain A


def stitch_predictions(chunk_results, chunk_ranges):
    """
    chunk_results: list of PDB text strings (one per chunk, in order)
    chunk_ranges: list of (start, end) residue index tuples matching chunk_results
    Returns: single merged PDB text string
    """
    ref_chain = _parse_pdb_text(chunk_results[0])
    merged_residues = list(ref_chain.get_residues())

    for i in range(1, len(chunk_results)):
        prev_start, prev_end = chunk_ranges[i - 1]
        cur_start, cur_end = chunk_ranges[i]
        overlap_len = prev_end - cur_start

        cur_chain = _parse_pdb_text(chunk_results[i])
        cur_residues = list(cur_chain.get_residues())

        # Overlapping residues: last `overlap_len` of merged_residues
        # vs first `overlap_len` of cur_residues
        fixed_atoms = [res["CA"] for res in merged_residues[-overlap_len:] if "CA" in res]
        moving_atoms = [res["CA"] for res in cur_residues[:overlap_len] if "CA" in res]

        n = min(len(fixed_atoms), len(moving_atoms))
        sup = Superimposer()
        sup.set_atoms(fixed_atoms[:n], moving_atoms[:n])
        sup.apply(cur_chain.get_atoms())  # rotate/translate whole chunk into global frame

        # Drop the overlapping residues from the new chunk, keep the rest
        merged_residues.extend(cur_residues[overlap_len:])

    # Rebuild a clean structure from merged residues
    structure = Structure("stitched")
    model = Model(0)
    chain = Chain("A")
    for idx, res in enumerate(merged_residues, start=1):
        res.id = (" ", idx, " ")
        chain.add(res)
    model.add(chain)
    structure.add(model)

    output = io.StringIO()
    io_writer = PDBIO()
    io_writer.set_structure(structure)
    io_writer.save(output)
    return output.getvalue()


def predict_structure_chunked(sequence: str, retries: int = 3) -> str:
    """Predict structure for sequences of any length by chunking + stitching."""
    if len(sequence) <= MAX_ESMFOLD_LENGTH:
        return predict_structure(sequence, retries=retries)

    chunks = chunk_sequence(sequence)
    chunk_results = []
    for i, (start, end, subseq) in enumerate(chunks):
        print(f"  chunk {i+1}/{len(chunks)}: residues {start}-{end}")
        pdb_text = predict_structure(subseq, retries=retries)
        chunk_results.append(pdb_text)

    chunk_ranges = [(s, e) for s, e, _ in chunks]
    return stitch_predictions(chunk_results, chunk_ranges)

def show_structure(pdb_id, chain_id=None):
    view = py3Dmol.view(query=f"pdb:{pdb_id}", width=700, height=450)
    if chain_id:
        view.setStyle({"chain": chain_id}, {"cartoon": {"color": "spectrum"}})
    else:
        view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addStyle({"hetflag": True}, {"stick": {"colorscheme": "orangeCarbon"}})
    view.zoomTo()
    components.html(view._make_html(), height=470)


def show_quaternary_structure(pdb_id):
    view = py3Dmol.view(query=f"pdb:{pdb_id}", width=700, height=450)
    view.setStyle({}, {"cartoon": {"colorscheme": "chain"}})
    view.addStyle({"hetflag": True}, {"stick": {"colorscheme": "orangeCarbon"}})
    view.zoomTo()
    components.html(view._make_html(), height=470)

ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
PDB_FETCH_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
PDB_SEQ_URL = "https://www.rcsb.org/fasta/entry/{pdb_id}"

# Fill this in with 15-25 PDB IDs, chosen deliberately for variety:
# different lengths, fold classes (all-alpha, all-beta, mixed), and some
# recent/less-common structures. Document WHY you picked each one in your
# research data book -- judges will ask.
 
PROTEIN_LIST = [
    "1UBQ",   # Ubiquitin - small, all-beta, classic benchmark
    "1LYZ",   # Lysozyme - small, mixed alpha/beta
    "6LYZ",
    "1CRN",   # Crambin - very small
    "4HHB",   # Hemoglobin subunit - all-alpha
    # Add more PDB IDs here to reach 15-25 total.
]


@dataclass
class BenchmarkResult:
    pdb_id: str
    sequence_length: int
    rmsd: float
    tm_score: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Fetching data
# ---------------------------------------------------------------------------
def fetch_experimental_pdb(pdb_id: str) -> str:
    """Download the experimental structure file from RCSB."""
    response = requests.get(PDB_FETCH_URL.format(pdb_id=pdb_id), timeout=30)
    response.raise_for_status()
    return response.text

@st.cache_data
def fetch_sequence(pdb_id: str) -> str:
    response = requests.get(PDB_SEQ_URL.format(pdb_id=pdb_id), timeout=30)
    response.raise_for_status()
    lines = response.text.strip().splitlines()
    sequence_lines: List[str] = []
    for line in lines[1:]:
        if line.startswith(">"):
            break
        sequence_lines.append(line.strip())
    return "".join(sequence_lines)



@st.cache_data
def predict_structure(sequence, retries=3):
    if len(sequence) > MAX_ESMFOLD_LENGTH:
        return predict_structure_chunked(sequence, retries=retries)
    for attempt in range(retries):
        try:
            response = requests.post(ESMFOLD_URL, data=sequence, timeout=180)
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            last_error = error
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ESMFold prediction failed after {retries} attempts: {last_error}")



# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def extract_ca_coordinates(pdb_text: str, chain: Optional[str] = None) -> np.ndarray:
    """
    Extract C-alpha atom coordinates in residue order.
    If `chain` is given, only that chain is used; otherwise the first
    chain encountered is used (important for multi-chain PDB entries).
    """
    coords = []
    target_chain = chain
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        this_chain = line[21].strip()
        if target_chain is None:
            target_chain = this_chain
        if this_chain != target_chain:
            continue
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords.append((x, y, z))
    return np.array(coords, dtype=float)


# ---------------------------------------------------------------------------
# Kabsch alignment + RMSD
# ---------------------------------------------------------------------------
def kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """
    Optimally superpose coords_b onto coords_a (equal length, paired by
    residue index) and return the RMSD after alignment. Centering alone,
    without rotation, is NOT a valid RMSD.
    """
    assert coords_a.shape == coords_b.shape, "Coordinate sets must be equal length"

    a_centered = coords_a - coords_a.mean(axis=0)
    b_centered = coords_b - coords_b.mean(axis=0)

    h = b_centered.T @ a_centered
    u, s, vt = np.linalg.svd(h)

    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1, 1, d])
    rotation = vt.T @ correction @ u.T

    b_aligned = (rotation @ b_centered.T).T

    diff = a_centered - b_aligned
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

def kabsch_transform(coords_a: np.ndarray, coords_b: np.ndarray):
    
     mean_a = coords_a.mean(axis=0)
     mean_b = coords_b.mean(axis=0)
     a_centered = coords_a - mean_a
     b_centered = coords_b - mean_b
     
     h = b_centered.T @ a_centered
     u, s, vt = np.linalg.svd(h)
     d = np.sign(np.linalg.det(vt.T @ u.T))
     correction = np.diag([1, 1, d])
     rotation = vt.T @ correction @ u.T

     return rotation, mean_a, mean_b


def apply_transform(coords: np.ndarray, rotation, mean_a, mean_b):
    return (rotation @ (coords - mean_b).T).T + mean_a


# ---------------------------------------------------------------------------
# TM-score
# ---------------------------------------------------------------------------
def tm_score(coords_a: np.ndarray, coords_b: np.ndarray, target_length: int) -> float:
    """
    Standard TM-score (Zhang & Skolnick, 2004) using the fixed d0 formula.
    coords_a, coords_b must already be paired by residue index.
    target_length is the length of the reference (experimental) structure.
    """
    assert coords_a.shape == coords_b.shape

    a_centered = coords_a - coords_a.mean(axis=0)
    b_centered = coords_b - coords_b.mean(axis=0)

    h = b_centered.T @ a_centered
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1, 1, d])
    rotation = vt.T @ correction @ u.T
    b_aligned = (rotation @ b_centered.T).T

    distances = np.sqrt(np.sum((a_centered - b_aligned) ** 2, axis=1))

    if target_length > 15:
        d0 = 1.24 * (target_length - 15) ** (1 / 3) - 1.8
    else:
        d0 = 0.5

    d0 = max(d0, 0.5)

    score_terms = 1.0 / (1.0 + (distances / d0) ** 2)
    return float(np.sum(score_terms) / target_length)

# ---------------------------------------------------------------------------
# Main benchmarking loop
# ---------------------------------------------------------------------------
def benchmark_one(pdb_id: str) -> BenchmarkResult:
     print(f"[{pdb_id}] fetching experimental structure...")
     experimental_pdb = fetch_experimental_pdb(pdb_id)
     experimental_coords = extract_ca_coordinates(experimental_pdb)

     print(f"[{pdb_id}] fetching sequence...")
     sequence = fetch_sequence(pdb_id)
     if not sequence:
        return BenchmarkResult(pdb_id, 0, float("nan"), float("nan"),
                                error="Could not retrieve sequence")

     print(f"[{pdb_id}] requesting ESMFold prediction ({len(sequence)} residues)...")
     predicted_pdb = predict_structure(sequence)
     predicted_coords = extract_ca_coordinates(predicted_pdb)

    # Experimental PDB files often have missing residues (crystal gaps).
    # Truncating to the shorter length is a simple safeguard, but a more
    # rigorous approach aligns by residue NUMBER, not just position.
    # Flag this as a known limitation in your methodology write-up.
     length = min(len(experimental_coords), len(predicted_coords))
     if length == 0:
        return BenchmarkResult(pdb_id, len(sequence), float("nan"), float("nan"),
                                error="No matching CA atoms found")

     exp_trimmed = experimental_coords[:length]
     pred_trimmed = predicted_coords[:length]

     rmsd = kabsch_rmsd(exp_trimmed, pred_trimmed)
     tm = tm_score(exp_trimmed, pred_trimmed, target_length=length)

     print(f"[{pdb_id}] RMSD = {rmsd:.2f} Å, TM-score = {tm:.3f}")
     return BenchmarkResult(pdb_id, len(sequence), rmsd, tm)

def benchmark_one_chain(pdb_id: str, chain: str) -> BenchmarkResult:
    print(f"[{pdb_id}:{chain}] fetching experimental structure...")
    experimental_pdb = fetch_experimental_pdb(pdb_id)
    experimental_coords = extract_ca_coordinates(experimental_pdb, chain=chain)

    print(f"[{pdb_id}:{chain}] fetching sequence...")
    sequence = fetch_sequence(pdb_id)
    if not sequence:
        return BenchmarkResult(pdb_id, 0, float("nan"), float("nan"),
                                error="Could not retrieve sequence")

    print(f"[{pdb_id}:{chain}] requesting ESMFold prediction ({len(sequence)} residues)...")
    predicted_pdb = predict_structure(sequence)
    predicted_coords = extract_ca_coordinates(predicted_pdb)

    length = min(len(experimental_coords), len(predicted_coords))
    if length == 0:
        return BenchmarkResult(pdb_id, len(sequence), float("nan"), float("nan"),
                                error="No matching CA atoms found")

    exp_trimmed = experimental_coords[:length]
    pred_trimmed = predicted_coords[:length]

    rmsd = kabsch_rmsd(exp_trimmed, pred_trimmed)
    tm = tm_score(exp_trimmed, pred_trimmed, target_length=length)

    print(f"[{pdb_id}:{chain}] RMSD = {rmsd:.2f} Å, TM-score = {tm:.3f}")
    return BenchmarkResult(pdb_id, len(sequence), rmsd, tm)


def get_chain_ids(pdb_id: str) -> list[str]:
    pdb_text = fetch_experimental_pdb(pdb_id)
    chains = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            c = line[21].strip()
            if c and c not in chains:
                chains.append(c)
    return chains

STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}

def is_protein_chain(pdb_text: str, chain: str) -> bool:
    """Check whether a chain is protein (has CA atoms with standard amino acid names)."""
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21].strip() != chain:
            continue
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        if atom_name == "CA" and res_name in STANDARD_AMINO_ACIDS:
            return True
    return False


def get_protein_chains(pdb_id: str) -> list[str]:
    """Return only protein chains for a PDB ID, skipping DNA/RNA chains."""
    experimental_pdb = fetch_experimental_pdb(pdb_id)
    chain_ids = get_protein_chains(pdb_id)
    chain_ids = get_protein_chains(pdb_id)
    print(f"DEBUG: protein chains found = {chain_ids}")
    protein_chains = [c for c in all_chains if is_protein_chain(experimental_pdb, c)]
    for c in set(all_chains) - set(protein_chains):
        print(f"[{pdb_id}] Skipping chain {c}: nucleic acid or non-protein, not scoreable by ESMFold.")
    return protein_chains


def benchmark_all_chains(pdb_id: str) -> list[BenchmarkResult]:
    chain_ids = get_chain_ids(pdb_id)
    results = []
    for c in chain_ids:
        try:
            results.append(benchmark_one_chain(pdb_id, c))
        except Exception as exc:
            print(f"[{pdb_id}:{c}] FAILED: {exc}")
    return results

def get_chain_prediction(pdb_id: str, chain: str):
    experimental_pdb = fetch_experimental_pdb(pdb_id)
    exp_coords = extract_ca_coordinates(experimental_pdb, chain=chain)
    sequence = fetch_sequence(pdb_id)
    predicted_pdb = predict_structure(sequence)
    pred_coords = extract_ca_coordinates(predicted_pdb)
    length = min(len(exp_coords), len(pred_coords))
    return exp_coords[:length], pred_coords[:length]


def benchmark_quaternary(pdb_id: str):
     chain_ids = get_chain_ids(pdb_id)
     if len(chain_ids) < 2:
        return {"error": f"{pdb_id} has only one chain — no quaternary structure to assess."}

     chain_data = {}
     for c in chain_ids:
        exp, pred = get_chain_prediction(pdb_id, c)
        chain_data[c] = {"exp": exp, "pred": pred}

     ref_chain = chain_ids[0]
     rotation, mean_a, mean_b = kabsch_transform(
        chain_data[ref_chain]["exp"], chain_data[ref_chain]["pred"]
     )

     exp_all, pred_all = [], []
     for c in chain_ids:
         exp_coords = chain_data[c]["exp"]
         pred_coords = chain_data[c]["pred"]
         if len(exp_coords) == 0 or len(pred_coords) == 0:
            continue
         exp_all.append(exp_coords)
         pred_all.append(apply_transform(pred_coords, rotation, mean_a, mean_b))

         if not exp_all:
             return {"error": f"{pdb_id} has no chains with valid matching coordinates."}

     exp_complex = np.vstack(exp_all)
     pred_complex = np.vstack(pred_all)
     complex_rmsd = float(np.sqrt(np.mean(np.sum((exp_complex - pred_complex) ** 2, axis=1))))

     return {
        "pdb_id": pdb_id,
        "chains": chain_ids,
        "reference_chain": ref_chain,
        "complex_rmsd": complex_rmsd,
        "exp_coords": exp_all,
        "pred_coords": pred_all,
     }

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
import io

def compute_sasa(pdb_text: str, chain: str = None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", io.StringIO(pdb_text))
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    sasa_by_residue = []
    for model in structure:
        for c in model:
            if chain and c.id != chain:
                continue
            for residue in c:
                if residue.has_id("CA"):
                    sasa_by_residue.append(residue.sasa)
        break
    return sasa_by_residue


def extract_plddt(pdb_text: str, chain: str = None):
    """ESMFold stores per-residue pLDDT in the B-factor column."""
    plddt = []
    seen_residues = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        this_chain = line[21].strip()
        if chain and this_chain != chain:
            continue
        res_num = line[22:26].strip()
        if res_num in seen_residues:
            continue
        seen_residues.add(res_num)
        bfactor = float(line[60:66])
        plddt.append(bfactor)
    return plddt


def find_epitope_candidates(pdb_id: str, chain: str, sasa_threshold=30.0, plddt_threshold=70.0):
    """Flag residues that are both surface-exposed AND confidently predicted."""
    sequence = fetch_sequence(pdb_id)
    predicted_pdb = predict_structure(sequence)
    sasa = compute_sasa(predicted_pdb, chain=None)  # predicted structure is single-chain
    plddt = extract_plddt(predicted_pdb, chain=None)

    length = min(len(sasa), len(plddt))
    candidates = []
    for i in range(length):
        if sasa[i] > sasa_threshold and plddt[i] > plddt_threshold:
            candidates.append({"residue_index": i + 1, "sasa": round(sasa[i], 1), "plddt": round(plddt[i], 1)})
    return candidates

def group_linear_epitopes(candidates: list, min_length: int = 5, max_gap: int = 1) -> list:
    """Group individual epitope-candidate residues into contiguous linear stretches."""
    if not candidates:
        return []
    indices = sorted(c["residue_index"] for c in candidates)
    groups = []
    current = [indices[0]]
    for idx in indices[1:]:
        if idx - current[-1] <= max_gap + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    return [
        {"start": g[0], "end": g[-1], "length": g[-1] - g[0] + 1}
        for g in groups if (g[-1] - g[0] + 1) >= min_length
    ]
def find_glycosylation_sites(sequence: str) -> list:
    """Find N-X-S/T sequon motifs (N-glycosylation sites), X != Proline."""
    sites = []
    for match in re.finditer(r"N[^P][ST]", sequence):
        sites.append(match.start() + 1)  # 1-indexed residue position
    return sites

def run_benchmark(pdb_ids: List[str], output_csv: str = "results.csv") -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
def _benchmark_one_safe(pdb_id: str) -> BenchmarkResult:
    try:
        result = benchmark_one(pdb_id)
    except Exception as exc:
        print(f"[{pdb_id}] FAILED: {exc}")
        result = BenchmarkResult(pdb_id, 0, float("nan"), float("nan"), error=str(exc))
    return result

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(_benchmark_one_safe, pdb_ids))

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pdb_id", "sequence_length", "rmsd_angstrom", "tm_score", "error"])
            for r in results:
                writer.writerow([r.pdb_id, r.sequence_length, r.rmsd, r.tm_score, r.error or ""])

        print(f"\nSaved {len(results)} results to {output_csv}")
        return results

def analyze_novel_sequence(sequence: str, sasa_threshold=30.0, plddt_threshold=70.0) -> dict:
    """Full pipeline for a sequence with no known experimental structure."""
    predicted_pdb = predict_structure(sequence)  # uses your existing chunking logic automatically
    sasa = compute_sasa(predicted_pdb, chain=None)
    plddt = extract_plddt(predicted_pdb, chain=None)

    length = min(len(sasa), len(plddt))
    candidates = []
    for i in range(length):
        if sasa[i] > sasa_threshold and plddt[i] > plddt_threshold:
            candidates.append({"residue_index": i + 1, "sasa": round(sasa[i], 1), "plddt": round(plddt[i], 1)})

    linear_epitopes = group_linear_epitopes(candidates)
    glyco_sites = find_glycosylation_sites(sequence)

    for epi in linear_epitopes:
        epi["glycan_overlap"] = any(epi["start"] <= g <= epi["end"] for g in glyco_sites)

    return {
        "predicted_pdb": predicted_pdb,
        "residue_candidates": candidates,
        "linear_epitopes": linear_epitopes,
        "glycosylation_sites": glyco_sites,
    }

if __name__ == "__main__":
    run_benchmark(PROTEIN_LIST)

import streamlit as st
import py3Dmol

st.title("ESMFold Benchmarking Pipeline")

pdb_ids = st.text_input("PDB IDs (comma-separated)", "1UBQ, 1LYZ, 6LYZ")

if st.button("Run Benchmark"):
     results = []
     for pdb_id in [p.rip() for p in pdb_ids.split(",")]:
         with st.spinner(f"Processing {pdb_id}..."):
             chain_results = benchmark_all_chains(pdb_id)
             for r in chain_results:
                 results.append({
                    "PDB ID": pdb_id,
                    "RMSD": r.rmsd,
                    "TM-score": r.tm_score,
                    "Error": r.error or ""
             })
             if len(chain_results) > 1:
                 st.subheader(f"{pdb_id} — Quaternary Structure Assessment")
                 quat_result = benchmark_quaternary(pdb_id)
                 if "error" in quat_result:
                     st.info(quat_result["error"])
                 else:
                     st.metric("Complex RMSD (Å)", f"{quat_result['complex_rmsd']:.2f}")
                     st.write(f"Reference chain used for alignment: **{quat_result['reference_chain']}**")
                     st.write(f"Chains assessed: {', '.join(quat_result['chains'])}")
                     st.subheader(f"{pdb_id} — Quaternary Structure")
                     show_quaternary_structure(pdb_id)
             else:
                 st.subheader(f"{pdb_id} — Predicted vs Experimental Structure")
                 show_structure(pdb_id)
     st.dataframe(results)
     
st.header("Novel Pathogen Epitope Triage")
st.caption("For sequences with no known experimental structure — predicts fold and flags likely surface-exposed, high-confidence epitope regions.")

raw_sequence = st.text_area("Paste amino acid sequence (FASTA sequence only, no header)")

if st.button("Analyze Sequence"):
    with st.spinner("Predicting structure and scanning for epitope candidates (this may take several minutes for large sequences)..."):
        analysis = analyze_novel_sequence(raw_sequence.strip())

    st.subheader(f"Found {len(analysis['linear_epitopes'])} candidate linear epitope regions")

    for epi in analysis["linear_epitopes"]:
        flag = " ⚠️ overlaps predicted glycosylation site" if epi["glycan_overlap"] else ""
        st.write(f"Residues {epi['start']}–{epi['end']} (length {epi['length']}){flag}")

    st.subheader("Predicted Structure")
    view = py3Dmol.view(width=700, height=450)
    view.addModel(analysis["predicted_pdb"], "pdb")
    view.setStyle({"cartoon": {"color": "lightgray"}})
    for epi in analysis["linear_epitopes"]:
        color = "orange" if epi["glycan_overlap"] else "red"
        view.addStyle({"resi": f"{epi['start']}-{epi['end']}"}, {"cartoon": {"color": color}})
    view.zoomTo()
    st.iframe(view._make_html(), height=470)
"""What a structure file says about itself, and what a viewer needs from it.

Two jobs, one rule. The jobs: turn a PDB, SDF or SMILES file into (label,
value) metadata rows for the UI and the knowledge base, and into the model text
3Dmol renders. The rule: **coordinate records are never shown as text.** A
10,000-line ATOM block is not a description of a protein, and indexed into the
knowledge base it would drown every real passage. Everything here reads the
header and counts what it has parsed; the only function that returns
coordinates is `viewer_model`, which feeds the viewer and nothing else.

Nothing in this module raises on bad input. A reader that fails becomes an
`("Error", reason)` row from `structure_metadata` and `None` from the viewer
functions, so a truncated download shows a message instead of a stack trace.
"""

from __future__ import annotations

import warnings

from biomentis.structures import format_for_path, inspect_structure

__all__ = [
    "depiction_png",
    "metadata_text",
    "sdf_record_count",
    "structure_metadata",
    "viewer_model",
]

_WATER = {"HOH", "WAT", "H2O", "DOD"}


def _error(reason: str) -> list[tuple[str, str]]:
    return [("Error", " ".join(str(reason).split()) or "unreadable structure file")]


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


# --- PDB ---------------------------------------------------------------------

def _record_text(lines: list[str], record: str) -> str:
    """The text of a (possibly continued) header record, joined with spaces."""
    return " ".join(ln[10:].strip() for ln in lines if ln.startswith(record)).strip()


def _key_values(text: str) -> list[tuple[str, str]]:
    """Split COMPND/SOURCE text (`KEY: value; KEY: value;`) into pairs."""
    pairs = []
    for chunk in text.split(";"):
        key, sep, value = chunk.partition(":")
        if sep and value.strip():
            pairs.append((key.strip().upper(), value.strip()))
    return pairs


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _hetnames(lines: list[str]) -> list[str]:
    """Ligand names from HETNAM, else the HETATM residue names; never water."""
    names: dict[str, str] = {}
    for ln in lines:
        if ln.startswith("HETNAM"):
            het_id, text = ln[11:14].strip(), ln[15:].strip()
            if ln[8:10].strip():  # continuation line
                names[het_id] = f"{names.get(het_id, '')} {text}".strip()
            else:
                names[het_id] = text
    out = [f"{name} ({het_id})" if name else het_id
           for het_id, name in names.items() if het_id not in _WATER]
    if out:
        return out
    return _unique(
        ln[17:20].strip() for ln in lines if ln.startswith("HETATM") and ln[17:20].strip() not in _WATER
    )


def _journal(lines: list[str]) -> str:
    fields: dict[str, str] = {}
    last = None
    for ln in lines:
        if not ln.startswith("JRNL"):
            continue
        sub, text = ln[12:16].strip(), ln[19:].strip()
        if sub:
            last = sub
        if last:
            fields[last] = " ".join(f"{fields.get(last, '')} {text}".split())
    parts = [fields.get("TITL", ""), fields.get("REF", "")]
    if fields.get("DOI"):
        parts.append("doi:" + fields["DOI"])
    return "; ".join(p for p in parts if p)


def _pdb_counts(path: str) -> tuple[int | None, int | None]:
    """(residues, atoms) in the first model. Waters and ligands are not residues."""
    from Bio.PDB import PDBParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PDBParser(QUIET=True).get_structure("s", path)[0]
    residues = sum(1 for r in model.get_residues() if r.id[0] == " ")
    atoms = sum(1 for _ in model.get_atoms())
    return residues or None, atoms or None


def _pdb_metadata(path: str) -> list[tuple[str, str]]:
    text = _read_text(path)
    lines = text.splitlines()
    if not any(ln.startswith(("ATOM", "HETATM")) for ln in lines):
        return _error("no atom records found in PDB file")

    header: dict = {}
    try:
        from Bio.PDB import parse_pdb_header

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            header = parse_pdb_header(path)
    except Exception:
        pass

    compnd = _key_values(_record_text(lines, "COMPND"))
    source = _key_values(_record_text(lines, "SOURCE"))
    stats = inspect_structure(path)
    try:
        residues, atoms = _pdb_counts(path)
    except Exception:
        residues, atoms = None, stats.n_atoms

    resolution = header.get("resolution")
    rows = [
        ("Title", _record_text(lines, "TITLE").capitalize() if _record_text(lines, "TITLE") else ""),
        ("Classification", (header.get("head") or "").strip().capitalize()),
        ("Deposited", header.get("deposition_date") or ""),
        ("Experimental method", _record_text(lines, "EXPDTA")),
        ("Resolution (Å)", f"{resolution:.2f}" if isinstance(resolution, (int, float)) else ""),
        ("Organism", ", ".join(_unique([v for k, v in source if k == "ORGANISM_SCIENTIFIC"]))),
        ("Expression system", ", ".join(_unique([v for k, v in source if k == "EXPRESSION_SYSTEM"]))),
        ("Molecule(s)", "; ".join(_unique([v for k, v in compnd if k == "MOLECULE"]))),
        ("Chains", ", ".join(stats.chains)),
        ("Residues", str(residues) if residues else ""),
        ("Atoms", str(atoms) if atoms else ""),
        ("Ligands / heteroatoms", ", ".join(_hetnames(lines))),
        ("Authors", ", ".join(a.strip() for a in _record_text(lines, "AUTHOR").split(",") if a.strip())),
        ("Reference", _journal(lines)),
        ("Keywords", _record_text(lines, "KEYWDS")),
    ]
    return [(k, v) for k, v in rows if v]


# --- SDF ---------------------------------------------------------------------

def _sdf_mols(path: str) -> list:
    from rdkit import Chem

    return list(Chem.SDMolSupplier(path, removeHs=False))


def sdf_record_count(path: str) -> int:
    """Number of records in an SDF file; 0 if it cannot be read."""
    try:
        return len(_sdf_mols(path))
    except Exception:
        return 0


def _formula_weight(mol) -> list[tuple[str, str]]:
    from rdkit.Chem import Descriptors, rdMolDescriptors

    return [
        ("Formula", rdMolDescriptors.CalcMolFormula(mol)),
        ("Molecular weight (g/mol)", f"{Descriptors.MolWt(mol):.2f}"),
    ]


def _sdf_metadata(path: str) -> list[tuple[str, str]]:
    mols = _sdf_mols(path)
    mol = mols[0] if mols else None
    if mol is None:
        return _error("no readable molecule in SDF file")
    name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
    rows = [("Name", name)] + _formula_weight(mol) + [
        ("Atoms", str(mol.GetNumAtoms())),
        ("Bonds", str(mol.GetNumBonds())),
    ]
    rows += [(tag, mol.GetProp(tag).strip()) for tag in mol.GetPropNames()]
    return [(k, v) for k, v in rows if v]


# --- SMILES ------------------------------------------------------------------

def _smiles_mol(path: str):
    """(mol, name) from a one-line .smi file; mol is None if it will not parse."""
    from rdkit import Chem

    first = next((ln.strip() for ln in _read_text(path).splitlines() if ln.strip()), "")
    parts = first.split(None, 1)
    if not parts:
        return None, ""
    return Chem.MolFromSmiles(parts[0]), parts[1].strip() if len(parts) == 2 else ""


def _smiles_metadata(path: str) -> list[tuple[str, str]]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    mol, name = _smiles_mol(path)
    if mol is None:
        return _error("not a valid SMILES string")
    rows = [("Name", name), ("Canonical SMILES", Chem.MolToSmiles(mol))] + _formula_weight(mol) + [
        ("H-bond donors", str(Lipinski.NumHDonors(mol))),
        ("H-bond acceptors", str(Lipinski.NumHAcceptors(mol))),
        ("Rotatable bonds", str(Lipinski.NumRotatableBonds(mol))),
        ("LogP", f"{Crippen.MolLogP(mol):.2f}"),
        ("TPSA (Å²)", f"{rdMolDescriptors.CalcTPSA(mol):.2f}"),
    ]
    return [(k, v) for k, v in rows if v]


# --- Public ------------------------------------------------------------------

def structure_metadata(path: str) -> list[tuple[str, str]]:
    """Ordered (label, value) rows describing the file. Never raises.

    Fields the file does not carry are omitted rather than shown empty or as
    "None". An unreadable file yields a single `("Error", reason)` row.
    """
    readers = {"pdb": _pdb_metadata, "sdf": _sdf_metadata, "smi": _smiles_metadata}
    reader = readers.get(format_for_path(path))
    if reader is None:
        return _error(f"unsupported structure format: {path}")
    try:
        return reader(path) or _error("no metadata found")
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}")


def metadata_text(path: str) -> str:
    """`label: value` lines from `structure_metadata`, for knowledge-base indexing."""
    return "\n".join(f"{label}: {value}" for label, value in structure_metadata(path))


def _embed_3d(mol):
    """A hydrogen-complete 3D copy of `mol`, or None if embedding fails."""
    from rdkit.Chem import AllChem

    mol = AllChem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass  # an unoptimized embedding still renders
    return mol


def viewer_model(path: str, record: int = 0) -> tuple[str, str] | None:
    """`(model_text, "pdb" | "sdf")` for 3Dmol, or None when it cannot be built.

    This is the one place coordinates leave this module, and only toward the
    viewer. SMILES has none, so it is embedded in 3D first.
    """
    from rdkit import Chem

    fmt = format_for_path(path)
    try:
        if fmt == "pdb":
            text = _read_text(path)
            if not any(ln.startswith(("ATOM", "HETATM")) for ln in text.splitlines()):
                return None
            return text, "pdb"
        if fmt == "sdf":
            mols = _sdf_mols(path)
            mol = mols[record] if 0 <= record < len(mols) else None
            return (Chem.MolToMolBlock(mol), "sdf") if mol is not None else None
        if fmt == "smi":
            mol, name = _smiles_mol(path)
            mol = _embed_3d(mol) if mol is not None else None
            if mol is None:
                return None
            if name:
                mol.SetProp("_Name", name)
            return Chem.MolToMolBlock(mol), "sdf"
    except Exception:
        return None
    return None


def depiction_png(path: str, record: int = 0) -> bytes | None:
    """A 2D drawing as PNG bytes for `.smi` / `.sdf`; None for PDB or on failure."""
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    fmt = format_for_path(path)
    try:
        if fmt == "smi":
            mol, _ = _smiles_mol(path)
        elif fmt == "sdf":
            mols = _sdf_mols(path)
            mol = mols[record] if 0 <= record < len(mols) else None
        else:
            return None
        if mol is None:
            return None
        mol = AllChem.RemoveHs(mol)
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None

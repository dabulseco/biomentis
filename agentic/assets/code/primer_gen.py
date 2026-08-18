import primer3
import re
from Bio import Entrez
from Bio import SeqIO

# ─── Helper Functions ───────────────────────────────────────────

def clean_sequence(seq):
    """Remove ambiguous bases and non-ACGT characters."""
    return re.sub(r'[^ACGT]', '', seq.upper())

def calculate_tm(seq):
    """Calculate Tm using primer3's nearest-neighbor model."""
    return primer3.calc_tm(seq)

def calculate_gc(seq):
    """Calculate GC content percentage."""
    return (seq.count('G') + seq.count('C')) / len(seq) * 100

def get_complement(seq):
    """Return reverse complement of a DNA sequence."""
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(comp[b] for b in reversed(seq))

# ─── Sequence Fetching ──────────────────────────────────────────

def fetch_bacterial_sequence(genus_species, gene="16S ribosomal RNA",
                             email="user@example.com"):
    """
    Fetch target DNA sequence from NCBI for a bacterial species.

    Parameters
    ----------
    genus_species : str
        e.g., "Escherichia coli", "Staphylococcus aureus"
    gene : str
        Target gene (default: 16S rRNA)
    email : str
        NCBI Entrez email (required)

    Returns
    -------
    seq : str
        DNA sequence
    desc : str
        Sequence description
    """
    Entrez.email = email

    # Search NCBI
    query = f'"{genus_species}"[Organism] AND "{gene}"[Title]'
    print(f"[INFO] Searching NCBI: {query}")

    handle = Entrez.esearch(db="nucleotide", term=query, retmax=10)
    record = Entrez.read(handle)
    handle.close()

    if not record["IdList"]:
        # Fallback: broader search
        query = f'"{genus_species}"[Organism] AND 16S[Title]'
        handle = Entrez.esearch(db="nucleotide", term=query, retmax=10)
        record = Entrez.read(handle)
        handle.close()

    if not record["IdList"]:
        raise ValueError(f"No sequences found for {genus_species}")

    # Prefer sequences 500-3000 bp (good LAMP targets)
    best_id = record["IdList"][0]
    best_len = 0
    for seq_id in record["IdList"][:5]:
        try:
            handle = Entrez.efetch(db="nucleotide", id=seq_id,
                                   rettype="gb", retmode="text")
            gb = SeqIO.read(handle, "genbank")
            handle.close()
            if 500 <= len(gb.seq) <= 3000 and len(gb.seq) > best_len:
                best_len = len(gb.seq)
                best_id = seq_id
        except:
            continue

    print(f"[INFO] Fetching sequence ID: {best_id}")
    handle = Entrez.efetch(db="nucleotide", id=best_id,
                           rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    handle.close()

    print(f"[INFO] {record.description[:100]}...")
    print(f"[INFO] Length: {len(record.seq)} bp")

    return str(record.seq), record.description

# ─── LAMP Primer Design Engine ───────────────────────────────────

def design_lamp_primers(target_seq, min_amplicon=120, max_amplicon=300):
    """
    Design a complete LAMP primer set for a target sequence.

    LAMP Primer Layout (5' → 3'):

        F3 →  F2 →  F1 →  [LOOP]  ← B1 ←  B2 ←  B3
        ─────────────────────────────────────────────
        F3: Forward outer primer
        F2: Forward inner primer (3' part of FIP)
        F1c: Forward inner complement (5' part of FIP)
        B1c: Backward inner complement (5' part of BIP)
        B2: Backward inner primer (3' part of BIP)
        B3: Backward outer primer

        FIP = F1c_complement + F2
        BIP = B1c_complement + B2

    Parameters
    ----------
    target_seq : str
        Target DNA sequence
    min_amplicon, max_amplicon : int
        Acceptable amplicon size range (F3 start to B3 end)

    Returns
    -------
    dict or None
        Primer design with keys: F3, B3, F2, F1c, B2, B1c, FIP, BIP,
        amplicon, score
    """
    seq = clean_sequence(target_seq)
    seq_len = len(seq)

    print(f"[INFO] Cleaned sequence: {seq_len} bp")
    print(f"[INFO] Amplicon range: {min_amplicon}-{max_amplicon} bp")

    # Design parameters
    params = {
        'F3':  {'len': (15,25), 'tm': (55,65), 'gc': (30,70)},
        'B3':  {'len': (15,25), 'tm': (55,65), 'gc': (30,70)},
        'F2':  {'len': (15,25), 'tm': (55,65), 'gc': (30,70)},
        'F1c': {'len': (15,25), 'tm': (60,70), 'gc': (30,70)},
        'B2':  {'len': (15,25), 'tm': (55,65), 'gc': (30,70)},
        'B1c': {'len': (15,25), 'tm': (60,70), 'gc': (30,70)},
    }

    def is_valid(seq, p):
        if not (p['len'][0] <= len(seq) <= p['len'][1]):
            return False
        tm = calculate_tm(seq)
        if not (p['tm'][0] <= tm <= p['tm'][1]):
            return False
        gc = calculate_gc(seq)
        if not (p['gc'][0] <= gc <= p['gc'][1]):
            return False
        # No homopolymer runs ≥5
        for base in 'ATGC':
            if base * 5 in seq:
                return False
        return True

    # Find F1c and B1c candidates (inner primers, higher Tm)
    f1c_list, b1c_list = [], []
    for ws in [18, 20, 22]:
        for i in range(seq_len - ws + 1):
            cand = seq[i:i+ws]
            if is_valid(cand, params['F1c']):
                f1c_list.append({'seq':cand, 'start':i, 'end':i+ws,
                                 'tm':calculate_tm(cand), 'gc':calculate_gc(cand)})
            if is_valid(cand, params['B1c']):
                b1c_list.append({'seq':cand, 'start':i, 'end':i+ws,
                                 'tm':calculate_tm(cand), 'gc':calculate_gc(cand)})

    print(f"[INFO] F1c: {len(f1c_list)}, B1c: {len(b1c_list)} candidates")

    f1c_list = f1c_list[:30]
    b1c_list = b1c_list[:30]

    best_score = float('inf')
    best = None

    for f1c in f1c_list:
        for b1c in b1c_list:
            if b1c['start'] <= f1c['end']:
                continue
            if not (10 <= b1c['start'] - f1c['end'] <= 100):
                continue

            # Find F2 (upstream of F1c)
            f2_list = []
            for ws in [18,20,22]:
                for i in range(max(0,f1c['start']-80), f1c['start']-ws+1):
                    cand = seq[i:i+ws]
                    if is_valid(cand, params['F2']):
                        f2_list.append({'seq':cand, 'start':i, 'end':i+ws,
                                        'tm':calculate_tm(cand), 'gc':calculate_gc(cand)})

            # Find B2 (downstream of B1c)
            b2_list = []
            for ws in [18,20,22]:
                for i in range(b1c['end'], min(seq_len, b1c['end']+80)-ws+1):
                    cand = seq[i:i+ws]
                    if is_valid(cand, params['B2']):
                        b2_list.append({'seq':cand, 'start':i, 'end':i+ws,
                                        'tm':calculate_tm(cand), 'gc':calculate_gc(cand)})

            if not f2_list or not b2_list:
                continue

            # Find F3 (outer forward)
            f3_list = []
            for ws in [18,20,22]:
                for i in range(max(0,f1c['start']-150), max(0,f1c['start']-40)-ws+1):
                    cand = seq[i:i+ws]
                    if is_valid(cand, params['F3']):
                        f3_list.append({'seq':cand, 'start':i, 'end':i+ws,
                                        'tm':calculate_tm(cand), 'gc':calculate_gc(cand)})

            # Find B3 (outer backward)
            b3_list = []
            for ws in [18,20,22]:
                for i in range(b1c['end']+40, min(seq_len, b1c['end']+150)-ws+1):
                    cand = seq[i:i+ws]
                    if is_valid(cand, params['B3']):
                        b3_list.append({'seq':cand, 'start':i, 'end':i+ws,
                                        'tm':calculate_tm(cand), 'gc':calculate_gc(cand)})

            if not f3_list or not b3_list:
                continue

            # Score all combinations
            for f2 in f2_list[:5]:
                for b2 in b2_list[:5]:
                    for f3 in f3_list[:5]:
                        for b3 in b3_list[:5]:
                            if not (f3['end'] <= f2['start'] <= f2['end'] <= f1c['start']):
                                continue
                            if not (b1c['end'] <= b2['start'] <= b2['end'] <= b3['start']):
                                continue

                            amp = b3['end'] - f3['start']
                            if not (min_amplicon <= amp <= max_amplicon):
                                continue

                            score = (abs(f3['tm']-60) + abs(b3['tm']-60) +
                                     abs(f2['tm']-60) + abs(b2['tm']-60) +
                                     abs(f1c['tm']-65) + abs(b1c['tm']-65))

                            if score < best_score:
                                best_score = score
                                best = {'F3':f3, 'B3':b3, 'F2':f2, 'F1c':f1c,
                                        'B2':b2, 'B1c':b1c, 'amplicon':amp, 'score':score}

    if best is None:
        return None

    # Assemble composite primers
    best['FIP'] = get_complement(best['F1c']['seq']) + best['F2']['seq']
    best['BIP'] = get_complement(best['B1c']['seq']) + best['B2']['seq']

    return best

# ─── Results Formatter ──────────────────────────────────────────

def print_lamp_results(design, species):
    """Pretty-print LAMP primer design results."""
    print("\n" + "="*70)
    print(f"  LAMP PRIMERS FOR {species}")
    print("="*70)
    print(f"  {'Primer':<6} {'Sequence':<22} {'Tm°C':<8} {'GC%':<7} {'Position'}")
    print(f"  {'-'*6} {'-'*22} {'-'*8} {'-'*7} {'-'*15}")
    for name in ['F3','B3','F2','F1c','B2','B1c']:
        p = design[name]
        print(f"  {name:<6} {p['seq']:<22} {p['tm']:<8.1f} {p['gc']:<7.1f} {p['start']}-{p['end']}")

    print(f"\n  Composite Primers:")
    print(f"  FIP: 5'-{get_complement(design['F1c']['seq'])} {design['F2']['seq']}-3'")
    print(f"  BIP: 5'-{get_complement(design['B1c']['seq'])} {design['B2']['seq']}-3'")
    print(f"\n  Amplicon: {design['amplicon']} bp | Score: {design['score']:.1f}")
    print("="*70)

# ─── Main Workflow ──────────────────────────────────────────────

def lamp_workflow(genus_species, gene="16S ribosomal RNA",
                  min_amp=120, max_amp=300, email="user@example.com"):
    """
    Complete LAMP primer design workflow.

    Usage:
        result = lamp_workflow("Escherichia coli")
        result = lamp_workflow("Staphylococcus aureus", gene="nuc")
    """
    print(f"\n{'█'*60}")
    print(f"█  LAMP DESIGN: {genus_species} | {gene}")
    print(f"{'█'*60}")

    # Step 1: Fetch sequence
    seq, desc = fetch_bacterial_sequence(genus_species, gene, email)

    # Step 2: Design primers
    design = design_lamp_primers(seq, min_amp, max_amp)

    if design is None:
        print("[FAIL] No valid primer set found. Try:")
        print("  - A different gene target")
        print("  - Relaxing amplicon size constraints")
        print("  - Using a longer input sequence")
        return None

    # Step 3: Display
    print_lamp_results(design, genus_species)

    design['species'] = genus_species
    design['gene'] = gene
    return design


# ─── Example Usage ──────────────────────────────────────────────

if __name__ == "__main__":
    # Design LAMP primers for E. coli 16S rRNA
    result = lamp_workflow("Escherichia coli")

    # For a different species, just change the name:
    # result = lamp_workflow("Salmonella enterica")
    # result = lamp_workflow("Listeria monocytogenes", gene="hlyA")

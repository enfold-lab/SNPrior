import numpy as np

import heapq
import os
from pathlib import Path
import subprocess

import pandas as pd
import pgenlib
from scipy import sparse



def LD_prunning(X, n_list, variant_info_df):
    """Training dosage matrix -> one int32 ranking score per SNP column.

    Pairwise Tagger best-N reimplementation, not ordinary LD pruning.
    Top max(n_list) tags receive unique scores n..1; all others receive 0.
    Inputs are assumed prevalidated.
    """
    assert X.shape[1] == len(variant_info_df), f"X has {X.shape[1]} SNP columns but variant_info_df has {len(variant_info_df)} rows"

    OUTPUT_DIR = "./results/LD_prunning"
    PLINK2_PATH = "plink2"
    MAF = 0.01
    GENO = 0.02
    WINDOW_KB = 200
    R2 = 1.0
    SEED = 42
    THREADS = 64
    MEMORY_MB = 8192

    n = int(max(n_list))
    samples, variants = X.shape
    plink = os.path.expanduser(os.fspath(PLINK2_PATH))
    out = Path(OUTPUT_DIR).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)
    prefix = work / "genotypes"

    # 1. Write PLINK sample and variant metadata without changing column order.
    ids = np.arange(1, samples + 1)
    pd.DataFrame({"#FID": ids, "IID": ids, "SEX": "NA"}).to_csv(prefix.with_suffix(".psam"), sep="\t", index=False)

    columns = ["CHROM", "POS", "REF", "ALT"]
    frame = variant_info_df[columns].copy()
    frame["CHROM"] = frame["CHROM"].str.removeprefix("chr").astype(np.int64)
    frame["POS"] = frame["POS"].astype(np.int64)
    frame["REF"] = frame["REF"].str.upper()
    frame["ALT"] = frame["ALT"].str.upper()
    frame.insert(2, "ID", np.arange(1, len(frame) + 1))
    frame.rename(columns={"CHROM": "#CHROM"}).to_csv(prefix.with_suffix(".pvar"), sep="\t", index=False)


    # 2. Convert all training SNPs to one PGEN file in 4096-SNP batches.
    with pgenlib.PgenWriter(os.fsencode(prefix.with_suffix(".pgen")), samples,
                           variant_ct=variants, nonref_flags=False) as writer:
        for start in range(0, variants, 4096):
            block = np.ascontiguousarray(X[:, start:start + 4096].T, dtype=np.int8)
            writer.append_biallelic_batch(block)
            del block
    del writer


    # 3. Training-only QC, then LD pairs within chromosome windows.
    command = [plink, "--pfile", str(prefix), "--threads", str(THREADS), "--memory", str(MEMORY_MB)]
    filters = ["--maf", str(MAF)] if MAF is not None else []
    filters += ["--geno", str(GENO)] if GENO is not None else []
    qc_prefix, ld_prefix = work / "qc", work / "ld"
    subprocess.run(command + filters + ["--mac", "1", "--write-snplist", "--out", str(qc_prefix)],
                   check=True, stdout=subprocess.DEVNULL)
    qc_file = qc_prefix.with_suffix(".snplist")
    qc = np.loadtxt(qc_file, dtype=np.int64, ndmin=1) - 1
    subprocess.run(command + ["--extract", str(qc_file), "--r2-unphased", "zs", "cols=id",
                              "--ld-window-kb", str(WINDOW_KB), "--ld-window", str(variants + 1),
                              "--ld-window-r2", str(R2), "--out", str(ld_prefix)],
                   check=True, stdout=subprocess.DEVNULL)


    # 4. Store qualifying LD connections sparsely, including self-coverage.
    lookup = np.full(variants, -1, dtype=np.int32)
    lookup[qc] = np.arange(len(qc), dtype=np.int32)
    edge_path, count = ld_prefix.with_suffix(".edges.bin"), 0
    decoder = subprocess.Popen([plink, "--zst-decompress", str(ld_prefix.with_suffix(".vcor.zst"))],
                               stdout=subprocess.PIPE)
    try:
        with edge_path.open("wb") as output, pd.read_csv(
            decoder.stdout, sep="\t", usecols=["#ID_A", "ID_B"], dtype=np.int64, chunksize=100_000,
        ) as chunks:
            for frame in chunks:
                pairs = frame[["#ID_A", "ID_B"]].to_numpy() - 1
                pairs = lookup[pairs]
                count += len(pairs)
                pairs.astype("<i4", copy=False).tofile(output)
                del frame, pairs
        if decoder.wait():
            raise RuntimeError("PLINK LD decompression failed")
    finally:
        decoder.stdout.close()
        if decoder.poll() is None:
            decoder.terminate()
        decoder.wait()
    del lookup, chunks, decoder
    if count:
        edges = np.memmap(edge_path, mode="r", dtype="<i4", shape=(count, 2))
        diagonal = np.arange(len(qc), dtype=np.int32)
        rows = np.concatenate((edges[:, 0], edges[:, 1], diagonal))
        cols = np.concatenate((edges[:, 1], edges[:, 0], diagonal))
        graph = sparse.coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                                  shape=(len(qc), len(qc))).tocsr()
        del edges, diagonal, rows, cols
    else:
        graph = sparse.eye(len(qc), format="csr", dtype=np.uint8)


    # 5. Greedy NEW coverage first; TOTAL coverage ranking of the full tag set.
    totals = np.diff(graph.indptr)
    priority = np.random.default_rng(SEED).permutation(len(totals))
    heap = [(-int(totals[i]), int(priority[i]), i) for i in range(len(totals))]
    heapq.heapify(heap)
    covered = np.zeros(len(totals), dtype=bool)
    tags, remaining = [], len(totals)
    while remaining:
        bound, tie, tag = heapq.heappop(heap)
        neighbors = graph.indices[graph.indptr[tag]:graph.indptr[tag + 1]]
        new = neighbors[~covered[neighbors]]
        if len(new) == 0:
            continue
        if len(new) != -bound:
            heapq.heappush(heap, (-len(new), tie, tag))
            continue
        tags.append(tag)
        covered[new] = True
        remaining -= len(new)
    if len(tags) < n:
        raise ValueError(f"Only {len(tags):,} tags available; cannot select {n:,}. No padding.")
    tags = np.asarray(tags, dtype=np.int64)
    order = np.lexsort((priority[tags], -totals[tags].astype(np.int64)))
    indices = qc[tags[order[:n]]]
    del graph, heap, covered, totals, priority, tags, order, neighbors, new


    # 6. Save original column indices and return unique scores for argsort.
    # np.save(out / "selected_indices.partial.npy", indices, allow_pickle=False)
    # (out / "selected_indices.partial.npy").replace(out / "selected_indices.npy")
    feature_importances_LD_prunning = np.zeros(variants, dtype=np.int32)
    feature_importances_LD_prunning[indices] = np.arange(n, 0, -1, dtype=np.int32)
    return feature_importances_LD_prunning


def select_feature(X, X_outer_train, variant_info_df, n_list):
    feature_importances_LD_prunning = LD_prunning(X_outer_train, n_list, variant_info_df)

    X_selected_list = []
    for n in n_list:
        selected_indices = np.argsort(feature_importances_LD_prunning)[-n:][::-1]
        X_selected = X[:, selected_indices]
        X_selected_list.append(X_selected)

    return X_selected_list


def main():
    pf = "/home/share/data/1kGP/preprocessed/merged_support3"
    n_pre_select_list = [100000, 1000000]

    # pf = "/home/share/data/1kGP/preprocessed/merged_support3_random_1k_seed_42"
    # n_pre_select_list = [100, 200]

    X = np.load(pf + "_matrix.npy")
    variant_info_df = pd.read_csv(pf + "_variant.csv", dtype = str)

    outer_train_idx = np.random.default_rng(seed=42).choice(X.shape[0], size=int(X.shape[0]*0.8),replace=False)
    X_outer_train = X[outer_train_idx]


    X_selected_list = select_feature(X, X_outer_train, variant_info_df, n_list=n_pre_select_list)

    for X_selected in X_selected_list:
        print(X_selected.shape)


if __name__ == "__main__":
    main()

"""Contains vcf scoring functions."""
import pandas as pd
import numpy as np
from .variant_processing import fetch_variant, geno_to_df, get_af_rec


def trim_alleles(ref, alt):
    """Left-align alleles by trimming common suffix first, then prefix"""
    # Remove common suffix (从右往左判断)
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref = ref[:-1]
        alt = alt[:-1]
    return alt

def normalize_variant(ref, used_alleles_list, effect_allele):
    """Normalize variant and calculate dosage based on effect_allele"""
    # 统计effect_allele在used_alleles_list中的出现次数
    dosage = 0
    for allele in used_alleles_list:
        # 对所有等位基因进行标准化处理（包括REF和ALT相同的情况）
        norm_allele = trim_alleles(ref, allele)
        if norm_allele[0] == effect_allele:
            dosage += 1
    return dosage

def score_geno(geno_df, variant_row, mode):
    contig_id = variant_row["contig_id"]
    var = f"{contig_id}:{variant_row['position']}"
    ref, effect_allele, beta = geno_df['REF'].iloc[0], variant_row['effect_allele'], variant_row['beta']
    geno_col_map = {v: i for i, v in enumerate(geno_df.iloc[0, 8].split(":"))}
    sample_data = geno_df.iloc[0, 9:].to_numpy(str)

    if mode == "GT" and "GT" in geno_col_map:
        alleles = [ref] + (geno_df['ALT'].iloc[0].split(',') if pd.notna(geno_df['ALT'].iloc[0]) else [])

        gt_data = np.array([entry.split(":")[geno_col_map["GT"]]
                           for entry in sample_data])
        # Normalize genotype separators
        gt_data = np.char.replace(gt_data, "|", "/")

        # 对每个基因型单独计算dosage和score
        scores = np.zeros(len(gt_data))
        for i, gt in enumerate(gt_data):
            current_alleles = []
            for allele_idx in gt.split('/'):
                if allele_idx.isdigit():
                    idx = int(allele_idx)
                    current_alleles.append(alleles[idx])
            # 对当前基因型计算dosage
            current_dosage = normalize_variant(ref, current_alleles, effect_allele)
            print(gt, ref, current_alleles, effect_allele)
            # 计算当前基因型的score
            scores[i] = beta * current_dosage
    elif mode == "GP" and "GP" in geno_col_map:
        gp_data = np.array([
            entry.split(":")[geno_col_map["GP"]]
            for entry in sample_data
        ])
        gp_probs = np.array([
            list(map(float, gp.split(","))) for gp in gp_data
        ])
        if gp_probs.shape[1] != 3:
            raise ValueError(
                "Invalid GP format: Expected exactly 3 probabilities per row.")
        dosage = 2 * gp_probs[:, 0] + gp_probs[:, 1]
        scores = beta * (dosage if effect_allele == ref else 2 - dosage)
    else:
        raise ValueError(
            f"Invalid mode '{mode}' or missing column '{mode}' in geno_col_map.")

    scores[np.isnan(scores)] = 0
    print(f"variant_row: {variant_row}, scores: {scores}, geno_df: {geno_df}")
    return pd.DataFrame({var: scores}, index=geno_df.columns[9:])


def impute_score_ref(samples, r, refbcf):
    """Impute missing variant scores using a reference VCF or a mapping file."""
    import os
    import pandas as pd
    import pysam
    import re

    contig = r['contig_id'].replace('chr', '')
    if refbcf.endswith(".txt"):
        if not os.path.exists(refbcf):
            raise FileNotFoundError(f"Mapping file not found: {refbcf}")
        # Load mapping file and standardize chromosome column
        mapping = pd.read_csv(
            refbcf, sep=r'\s+', header=None, names=["file", "chr"])
        mapping['chr'] = mapping['chr'].astype(
            str).str.replace('chr', '', regex=False)
        vcf_row = mapping[mapping['chr'] == contig]
        if vcf_row.empty:
            raise ValueError(f"No file for {r['contig_id']} in mapping.")
        if len(vcf_row) > 1:
            raise ValueError(
                f"Multiple files found for {r['contig_id']} in mapping.")
        vcf_path = vcf_row['file'].iloc[0]
    else:
        if not re.match(
            r".*\.vcf(\..*)?$|.*\.bcf(\..*)?$",
            refbcf,
                re.IGNORECASE):
            raise ValueError(
                "Invalid file format. Provide a .vcf/.bcf file or valid mapping.")
        vcf_path = refbcf
    if not os.path.exists(vcf_path):
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    with pysam.VariantFile(vcf_path, 'r') as var_obj:
        var = fetch_variant(var_obj, r, ["blank"])
        if not var:
            raise ValueError(
                f"SNP {r['contig_id']}:{r['position']} not found.")
    ref_df = geno_to_df(var, ["blank"])
    ref, alt, af = ref_df['REF'].iloc[0], ref_df['ALT'].iloc[0], get_af_rec(
        ref_df)
    if r['effect_allele'] not in {ref, alt}:
        raise ValueError("Alleles don't match.")
    af = 1 - af if r['effect_allele'] == ref else af
    return pd.DataFrame({f"{r['contig_id']}:{r['position']}_imputed": [
                        r['beta'] * af * (2 - af)]}, index=samples)

def impute_score_wildtype(samples, r, ref):
    import pandas as pd
    import pysam
    """从参考基因组获取野生型基因型进行imputation打分"""
    # 打开参考基因组文件
    with pysam.FastaFile(ref) as fasta:
        try:
            # 获取该位点的参考基因组碱基
            wildtype = fasta.fetch(r['contig_id'], r['position']-1, r['position'])
        except KeyError:
            # 处理染色体命名不一致的情况
            chrom = r['contig_id'].replace('chr', '')
            try:
                wildtype = fasta.fetch(chrom, r['position']-1, r['position'])
            except KeyError:
                raise ValueError(f"无法在参考基因组中找到位点 {r['contig_id']}:{r['position']}")

    # 判断effect_allele是否为野生型
    is_effect_wildtype = (r['effect_allele'].upper() == wildtype.upper())

    # 计算得分
    # 如果effect_allele是野生型,则基因型0/0对应剂量为2
    # 如果effect_allele不是野生型,则基因型0/0对应剂量为0
    dosage = 2 if is_effect_wildtype else 0
    score = r['beta'] * dosage
    print(f"effect_allele: {r['effect_allele']}, wildtype: {wildtype}, is_effect_wildtype: {is_effect_wildtype}, dosage: {dosage}, score: {score}")
    # 返回与原函数相同格式的DataFrame
    return pd.DataFrame({
        f"{r['contig_id']}:{r['position']}_ref": [score]
    }, index=samples)

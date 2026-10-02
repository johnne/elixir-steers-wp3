#!/usr/bin/env python
import pandas as pd, numpy as np


def benchmark2energytab(
    benchmark_f,
    threads_f,
    per_rule_sample_outfile,
    energy_row_outfile,
    W_CORE=3.5,
    W_GB=0.3725,
    PUE=1.15,
    MB_PER_CORE=1776,
):
    """Energy / carbon estimate per job, legacy vs updated pipeline on Dardel.

    Inputs : benchmarks_tidy.csv (parsed Snakemake benchmarks, relaxed mmseqs stats already merged)
            threads.csv         (requested threads + pipeline part per rule)
    Outputs: energy_rows.csv            (one row per job)
            energy_per_rule_sample.csv (aggregated per version / part / rule / sample)

    Method (agreed basis):
      cores   = requested threads, or actual busy cores (cpu_time / s) if higher
      memory  = peak RAM (max_rss), i.e. optimal memory request
      energy  = runtime_h * (cores * W_CORE + mem_GB * W_GB) * PUE
      Wh_dardel_accounting: variant where memory also reserves cores (ceil(max_rss / MB_PER_CORE))
      QC rules are excluded (not present in both versions).

    Parameters
    ----------
    benchmark_f : benchmarks_tidy.csv (parsed Snakemake benchmarks, relaxed mmseqs stats already merged)
    threads_f : threads.csv (requested threads + pipeline part per rule)
    per_rule_sample_outfile : write results aggregated to rule/sample here ("energy_per_rule_sample.csv" originally)
    energy_row_outfile : write results for energy per row here ("energy_rows.csv" originally)
    W_CORE : W per core
    W_GB : W per GB RAM
    PUE : total facility energy ÷ energy used by the IT equipment. 1.15 as an
    assumed value for Dardel: it's liquid-cooled, which is much more efficient
    than air cooling, and sits in a cool climate. It isn't a published figure,
    so PDC is the source to ask for a real number. It doesn't capture heat
    reuse. Dardel's waste heat goes into KTH's heating system, which PUE
    ignores. A related metric, ERE (energy reuse effectiveness), subtracts the
    reused energy, and for Dardel it could come out below 1.0. Whether to credit
    that is a judgment call. Most footprint calculations, including Green
    Algorithms, just use PUE, which is the more conservative choice.
    MB_PER_CORE : Memory per core (MB)
    """
    # W_CORE, W_GB, PUE, MB_PER_CORE = 3.5, 0.3725, 1.15, 1776   # W per core, W per GB RAM, PUE, Dardel MB per core

    d = pd.read_csv(benchmark_f).merge(
        pd.read_csv(threads_f), on=["version", "rule"], how="left"
    )
    assert d.threads.notna().all(), "rule missing from threads.csv"
    d = d[d.pipeline_part != "qc"].copy()

    h = d.s / 3600
    d["busy_cores"] = (d.cpu_time / d.s.replace(0, np.nan)).fillna(0)
    d["cores"] = np.maximum(d.threads, d.busy_cores)
    d["mem_GB"] = d.max_rss / 1000
    d["core_h"] = h * d.cores
    d["Wh_cpu"] = h * d.cores * W_CORE * PUE
    d["Wh_mem"] = h * d.mem_GB * W_GB * PUE
    d["Wh_total"] = d.Wh_cpu + d.Wh_mem
    d["cores_dardel_accounting"] = np.maximum(d.cores, np.ceil(d.max_rss / MB_PER_CORE))
    d["Wh_dardel_accounting"] = (
        h * (d.cores_dardel_accounting * W_CORE + d.mem_GB * W_GB) * PUE
    )

    rows = d[
        [
            "version",
            "pipeline_part",
            "rule",
            "sample",
            "split",
            "path",
            "threads",
            "s",
            "cpu_time",
            "max_rss",
            "busy_cores",
            "cores",
            "core_h",
            "Wh_cpu",
            "Wh_mem",
            "Wh_total",
            "cores_dardel_accounting",
            "Wh_dardel_accounting",
        ]
    ]
    if energy_row_outfile is not None:
        rows.round(6).to_csv(energy_row_outfile, index=False)

    agg = (
        d.groupby(["version", "pipeline_part", "rule", "sample"])
        .agg(
            threads=("threads", "first"),
            n_jobs=("s", "size"),
            runtime_s=("s", "sum"),
            cpu_time_s=("cpu_time", "sum"),
            peak_rss_MB=("max_rss", "max"),
            core_h=("core_h", "sum"),
            Wh_cpu=("Wh_cpu", "sum"),
            Wh_mem=("Wh_mem", "sum"),
            Wh_total=("Wh_total", "sum"),
            Wh_dardel_accounting=("Wh_dardel_accounting", "sum"),
        )
        .reset_index()
    )
    if per_rule_sample_outfile is not None:
        agg.round(4).to_csv(per_rule_sample_outfile, index=False)

    tot = agg.groupby("version")[
        ["core_h", "Wh_cpu", "Wh_mem", "Wh_total", "Wh_dardel_accounting"]
    ].sum()
    print(tot.round(1))
    print("updated / legacy:")
    print((tot.loc["updated"] / tot.loc["legacy"]).round(3))

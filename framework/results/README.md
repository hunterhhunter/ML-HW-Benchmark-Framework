# Benchmark results

`benchmark_results.legacy.csv` is an immutable, structurally inconsistent
historical archive. It is preserved byte-for-byte for reference, is not read by
the benchmark framework by default, and must not be automatically padded or
truncated.

The first benchmark execution creates a new `benchmark_results.csv` using the
current result schema. Generated result CSVs and their details, traces, and run
reservation artifacts are intentionally untracked.

# Benchmark results

Historical results belong in local archives outside Git. The benchmark
framework does not read older schemas by default or automatically alter their
columns.

The first benchmark execution creates a new `benchmark_results.csv` using the
current result schema. Generated result CSVs and their details, traces, and run
reservation artifacts are intentionally untracked.

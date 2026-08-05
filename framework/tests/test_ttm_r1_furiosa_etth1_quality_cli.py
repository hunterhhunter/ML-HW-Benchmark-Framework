import runpy


def test_quality_cli_accepts_fixed_local_paths_and_defaults_to_240_windows():
    """Catches a runbook command being unable to express the fixed quality contract."""
    module = runpy.run_path(
        "framework/tools/ttm_r1_furiosa_etth1_quality.py", run_name="not_main"
    )

    args = module["build_parser"]().parse_args(
        [
            "--model-path",
            "/models/ttm",
            "--dataset-path",
            "/data/ETTh1.csv",
            "--output-dir",
            "/results/out",
        ]
    )

    assert str(args.model_path) == "/models/ttm"
    assert str(args.dataset_path) == "/data/ETTh1.csv"
    assert str(args.output_dir) == "/results/out"
    assert args.windows == 240
    assert args.strict_parity_result is None

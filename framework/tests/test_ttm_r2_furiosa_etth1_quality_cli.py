import runpy


def test_r2_quality_cli_defaults_to_240_fixed_etth1_windows():
    module = runpy.run_path("framework/tools/ttm_r2_furiosa_etth1_quality.py", run_name="not_main")
    args = module["build_parser"]().parse_args([
        "--model-path", "/models/r2", "--dataset-path", "/data/ETTh1.csv", "--output-dir", "/results/out",
    ])
    assert args.windows == 240
    assert args.strict_parity_result is None

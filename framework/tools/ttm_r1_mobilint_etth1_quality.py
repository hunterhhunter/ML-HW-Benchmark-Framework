#!/usr/bin/env python3
"""Run train-calibrated TTM-R1 MXQ quality evaluation on remote ARIES."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(root), str(root / "src")]
from chronos_bolt.evidence import write_result
from ttm_r1.core import TTMR1Core, load_ttm_r1_model
from ttm_r1.etth1_quality import ETTh1QualityConfig, evaluate_prepared_windows, load_etth1_windows, percentage_degradation
from ttm_r1.host_adapter import TTMR1HostAdapter
from ttm_r1.mobilint_aries import quantize_core_input, restore_artifact_output

def build_parser():
    p=argparse.ArgumentParser(description="Measure calibrated TTM-R1 MXQ ETTh1 quality on ARIES")
    for name in ("model-path","dataset-path","artifact","output-dir"): p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--compile-result",type=Path); p.add_argument("--windows",type=int,default=240); return p

def build_aries_runner(model):
    input_shape=tuple(model.get_model_input_shape()[0]); output_shape=tuple(model.get_model_output_shape()[0]); scale=model.get_input_scale()[0]
    def run(core_input):
        value,saturated=quantize_core_input(core_input,input_shape,scale)
        raw=model.infer_to_float([value])[0]
        return restore_artifact_output(raw,output_shape),saturated
    return run

def describe_scale(scale):
    """Serialize only the documented scalar/list fields of a qbruntime Scale."""
    return {"scale":float(scale.scale),"is_uniform":bool(scale.is_uniform),"scale_list":[float(x) for x in scale.scale_list],"zero_point":int(scale.zero_point),"is_asymmetric":bool(scale.is_asymmetric),"zero_points":[int(x) for x in scale.zero_points]}

def run(args):
    import qbruntime
    if 0 not in qbruntime.get_available_device_numbers(): raise RuntimeError("ARIES device 0 is unavailable")
    if not args.artifact.is_file(): raise ValueError(f"MXQ artifact is missing: {args.artifact}")
    model=load_ttm_r1_model(str(args.model_path)); core=TTMR1Core(model).eval(); adapter=TTMR1HostAdapter(core.contract,split_ttm_scaler=True)
    contexts,targets,split=load_etth1_windows(ETTh1QualityConfig(args.dataset_path,windows=args.windows))
    runtime=qbruntime.load(str(args.artifact)); runner=build_aries_runner(runtime); saturation=0
    runtime_abi={"input_shape":list(runtime.get_model_input_shape()[0]),"output_shape":list(runtime.get_model_output_shape()[0]),"input_scale":describe_scale(runtime.get_input_scale()[0])}
    def device(inputs):
        nonlocal saturation
        out,count=runner(inputs[0].detach().cpu().numpy()); saturation+=count; return torch.from_numpy(out)
    try: quality=evaluate_prepared_windows(core,adapter,contexts,targets,device)
    finally: runtime.dispose()
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True); np.savez_compressed(out/"mobilint-etth1-quality-predictions.npz",cpu_predictions=quality["cpu_predictions"].numpy(),aries_predictions=quality["rngd_predictions"].numpy(),targets=targets.numpy())
    result={"status":"measured","vendor":"mobilint","runtime_success":True,"task_quality_status":"measured","quantization_status":"saturated" if saturation else "unsaturated","saturation":{"elements":saturation,"total":args.windows*512},"artifact":str(args.artifact.resolve()),"runtime_abi":runtime_abi,"dataset":{"path":str(args.dataset_path.resolve()),"column":"OT","split":split},"cpu_task":quality["cpu_task"],"aries_task":quality["rngd_task"],"prediction_delta":quality["prediction_delta"],"degradation_percent":{n:percentage_degradation(quality["cpu_task"][n],quality["rngd_task"][n]) for n in ("mae","rmse")}}
    if args.compile_result: result["compile_result"]=str(args.compile_result.resolve())
    return write_result(out/"mobilint-etth1-quality-result.json",result)

def main(argv:Sequence[str]|None=None): print(run(build_parser().parse_args(argv))); return 0
if __name__=="__main__": raise SystemExit(main())

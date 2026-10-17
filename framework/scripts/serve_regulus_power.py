#!/usr/bin/env python3
"""Sample REGULUS 12 V DC input on a Jetson-connected INA3221.

The sensor worker runs only during a requested benchmark interval. Run this
server on the Jetson before starting run_regulus_power.py on REGULUS.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path

from regulus_power_link import InaWorkerSampler, PowerServer


def _ipv4(value: str) -> str:
    try:
        return str(ipaddress.IPv4Address(value))
    except ipaddress.AddressValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", type=_ipv4, required=True,
                        help="Jetson LAN address reachable from REGULUS")
    parser.add_argument("--allow-client", type=_ipv4, required=True,
                        help="REGULUS LAN address allowed to request a run")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bus", type=int, default=7)
    parser.add_argument("--address", type=lambda value: int(value, 0), default=0x40)
    parser.add_argument("--shunt-ohms", type=float, default=0.05)
    parser.add_argument("--max-current-a", type=float, default=3.0)
    parser.add_argument("--min-bus-v", type=float, default=10.0)
    parser.add_argument("--max-bus-v", type=float, default=14.0)
    parser.add_argument("--sample-interval-sec", type=float, default=0.1)
    parser.add_argument("--worker-path", type=Path,
                        help="INA3221 worker file if the Jetson has only deployed scripts")
    parser.add_argument("--check-sensor", action="store_true",
                        help="read one sample and exit without listening")
    return parser


def make_server(args: argparse.Namespace) -> PowerServer:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in 1..65535")
    if args.bus < 0 or not 0x03 <= args.address <= 0x77:
        raise ValueError("invalid I2C bus or address")
    if args.shunt_ohms <= 0 or not 0 < args.max_current_a <= 3.2:
        raise ValueError("invalid shunt or current limit")
    worker = (args.worker_path if args.worker_path is not None else
              Path(__file__).resolve().parent.parent / "src" / "monitors" / "ina3221_worker.py")
    if not worker.is_file():
        raise FileNotFoundError(worker)
    sampler = InaWorkerSampler(
        worker, bus=args.bus, address=args.address,
        shunt_ohms=args.shunt_ohms, max_current_a=args.max_current_a,
    )
    return PowerServer(
        host=args.bind, port=args.port, output_root=args.output_root,
        sample_interval_sec=args.sample_interval_sec,
        sample_fn=sampler, max_current_a=args.max_current_a,
        allowed_client_ip=args.allow_client,
        min_bus_v=args.min_bus_v, max_bus_v=args.max_bus_v,
    )


def main() -> int:
    args = build_parser().parse_args()
    server = make_server(args)
    sampler = server.sample_fn
    sampler.start()
    try:
        bus_v, current_a, power_w = sampler()
    finally:
        sampler.close()
    if not server.min_bus_v <= bus_v <= server.max_bus_v:
        raise RuntimeError(
            f"REGULUS bus voltage {bus_v:.3f} V is outside "
            f"{server.min_bus_v:.3f}..{server.max_bus_v:.3f} V; check DC ground wiring"
        )
    print(json.dumps({"sensor": "INA3221", "bus_V": bus_v,
                      "current_A": current_a, "power_W": power_w}), flush=True)
    if args.check_sensor:
        return 0
    print(f"listening on {args.bind}:{args.port}; allow-client={args.allow_client}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

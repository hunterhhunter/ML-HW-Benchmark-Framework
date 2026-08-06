"""CPU scaling boundary for TTM-R2's fixed 512-96 contract."""

from __future__ import annotations

from ttm_r1.host_adapter import PreparedTTMR1Inputs

from .contracts import TTMR2Contract


class TTMR2HostAdapter:
    """Delegate numerically identical standard scaling to the proven R1 adapter."""

    def __init__(self, contract: TTMR2Contract | None = None, *, split_ttm_scaler: bool = True) -> None:
        from ttm_r1.host_adapter import TTMR1HostAdapter

        self.contract = contract or TTMR2Contract.fixed()
        self._adapter = TTMR1HostAdapter(split_ttm_scaler=split_ttm_scaler)

    def prepare(self, context):
        return self._adapter.prepare(context)


PreparedTTMR2Inputs = PreparedTTMR1Inputs

"""Connectome workflow logic glue for SlicerDSpace."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from connectome.pipeline import ConnectomeConfig, ConnectomeResult, run_connectome


class ConnectomeLogicMixin:
    """Logic entrypoints for connectome-related workflows."""

    def runConnectomePipeline(
        self,
        *,
        subject_id: str,
        tractography_path: Path,
        parcellation_path: Path,
        output_dir: Optional[Path] = None,
        metric: str = "streamline_count",
        node_source: str = "ddparcel",
        compress_mrtrix_dense: Optional[bool] = None,
        execute: bool = False,
    ) -> ConnectomeResult:
        config = ConnectomeConfig(
            subject_id=subject_id,
            tractography_path=tractography_path,
            parcellation_path=parcellation_path,
            output_dir=output_dir,
            metric=metric,
            node_source=node_source,
            compress_mrtrix_dense=compress_mrtrix_dense,
            execute=execute,
        )
        return run_connectome(config, log=self.log)

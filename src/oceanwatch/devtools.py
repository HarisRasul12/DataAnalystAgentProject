from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def serve_main() -> None:
    cmd = [
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8000",
        "--server.headless",
        "true",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=Path.cwd()))


def test_main() -> None:
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "tests", "-v"], cwd=Path.cwd()))


def smoke_main() -> None:
    command = (
        "from oceanwatch.config import load_settings; "
        "from oceanwatch.service import OceanWatchService; "
        "from oceanwatch.schemas import AnalysisRequest; "
        "settings=load_settings(); "
        "settings=settings.__class__(**{**settings.__dict__, 'adk_enabled': False, 'require_adk_success': False}); "
        "svc=OceanWatchService(settings=settings); "
        "req=AnalysisRequest(user_question='Quick smoke run for OceanWatch analytics.', station_key='sf_bay', hours_back=24); "
        "res=svc.run_analysis(req); "
        "print('thesis:', res.insight.thesis[:120]); "
        "print('health:', res.health)"
    )
    raise SystemExit(subprocess.call([sys.executable, "-c", command], cwd=Path.cwd()))

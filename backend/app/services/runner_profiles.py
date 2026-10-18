"""한 호스트에 설치된 여러 벤더 Python 환경의 서버 측 등록 정보."""

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from .benchmark_service import FRAMEWORK_DIR, FRAMEWORK_PYTHON


@dataclass(frozen=True)
class RunnerProfile:
    id: str
    python: Path
    targets: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def matches(self, target_id: str | None, model: str) -> bool:
        return (not self.targets or target_id in self.targets) and (
            not self.models or model in self.models
        )

    def score(self) -> int:
        return int(bool(self.targets)) + int(bool(self.models))

    def fingerprint(self) -> str:
        payload = {
            "id": self.id, "python": str(self.python), "targets": self.targets,
            "models": self.models, "env": self.env,
        }
        return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def public(self) -> dict:
        return {
            "id": self.id,
            "python": str(self.python),
            "targets": list(self.targets),
            "models": list(self.models),
            "fingerprint": self.fingerprint(),
            "available": self.python.is_file() and os.access(self.python, os.X_OK),
        }


def load_runner_profiles() -> list[RunnerProfile]:
    config_path = os.getenv("WEB_RUNNERS_FILE")
    if not config_path:
        python = FRAMEWORK_PYTHON if FRAMEWORK_PYTHON.is_file() else Path(sys.executable)
        return [RunnerProfile(id="default", python=python)]
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    profiles: list[RunnerProfile] = []
    seen: set[str] = set()
    for item in raw.get("profiles", []):
        profile_id = item["id"]
        python = Path(item["python"])
        if not profile_id or profile_id in seen or not python.is_absolute():
            raise ValueError("runner profile의 id는 고유해야 하고 python은 절대 경로여야 합니다")
        seen.add(profile_id)
        profiles.append(RunnerProfile(
            id=profile_id,
            python=python,
            targets=tuple(item.get("targets", [])),
            models=tuple(item.get("models", [])),
            env=dict(item.get("env", {})),
        ))
    if not profiles:
        raise ValueError("등록된 runner profile이 없습니다")
    return profiles


def select_runner_profile(
    target_id: str | None,
    model: str,
    selected_id: str | None = None,
) -> RunnerProfile:
    profiles = load_runner_profiles()
    matches = [profile for profile in profiles if profile.matches(target_id, model)]
    if selected_id:
        matches = [profile for profile in matches if profile.id == selected_id]
    if not matches:
        raise ValueError("target/model에 맞는 등록된 Python 실행 환경이 없습니다")
    profile = max(matches, key=RunnerProfile.score)
    if not profile.python.is_file() or not os.access(profile.python, os.X_OK):
        raise ValueError(f"Python 실행 파일을 사용할 수 없습니다: {profile.id}")
    if not (FRAMEWORK_DIR / "src" / "main.py").is_file():
        raise ValueError("프레임워크 CLI 파일을 찾을 수 없습니다")
    return profile

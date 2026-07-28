from dataclasses import dataclass
from pathlib import Path
from .model_spec import Model_Spec

@dataclass(frozen=True)
class CompiledModel:
    """
    컴파일이 완료된 모델의 결과를 담는 불변 컨테이너
    Runtime 모듈은 이 객체만을 입력받아 의존성을 분리하고 타입 안정성을 보장

    artifact_path가 None이면 런타임이 모델 참조로부터 실행 artifact를 해석한다.
    """
    spec: Model_Spec        # 원본 설계도 명세
    backend_name: str       # 실행 타겟 백엔드 이름 (예: 'iree-cpu', 'onnx-cuda')
    artifact_path: Path | None  # 컴파일된 바이너리 경로 또는 런타임 해석 상태

    def __post_init__(self):
        """
        객체가 생성되는 시점(즉, 컴파일러가 반환하는 시점)에 
        명시된 artifact의 실제 파일 존재 여부를 1차로 강제 검증
        이렇게 하면 Runtime 클래스 내부에서는 파일 존재 여부를 체크할 필요가 없음
        """
        if self.artifact_path is not None and not self.artifact_path.exists():
            raise FileNotFoundError(
                f"[Compiler Data Error] The compiled artifact does not exist at '{self.artifact_path}'. "
                "The compiler must ensure the binary is successfully generated before returning CompiledModel."
            )

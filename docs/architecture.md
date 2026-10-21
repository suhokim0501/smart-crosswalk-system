# 함수별 역할과 계산 기준

## 실행과 자료 흐름

`cli.main()` → `config.load_config()` → `pipeline.run_analysis()` → `reporting.write_report()`.
`demo`는 영상·모델 대신 `demo.run_demo()`에서 생성한 경로를 사용합니다.

| 모듈 | 주요 함수·클래스 | 역할 |
|---|---|---|
| `cli` | `build_parser`, `main` | 명령 해석, 설정 덮어쓰기, 오류 메시지·종료 코드 |
| `config` | `load_config`, `CrosswalkConfig`, `TrackingConfig`, `ClassificationConfig` | TOML 로딩, 상대 경로 해석, 차원·시간·범위 검증 |
| `types` | `Detection`, `Observation`, `TrackDecision`, `CrossingDecision` | 단계 사이에 전달하는 값과 단위 표현 |
| `video` | `inspect_video` | FPS·해상도·프레임 수 확인 |
| `video` | `iter_frames` | 소스 프레임 번호와 실제 시간 보존, stride 적용, 종료 시 자원 해제 |
| `video` | `crop_bbox` | 이미지 경계 안에서 분류용 바운딩 박스 잘라내기 |
| `backend` | `YoloBackend.track` | 동일한 크기의 프레임으로 YOLO·ByteTrack 실행, ID·박스 반환 |
| `backend` | `YoloBackend.predict` | crop 목록의 분류 확률 반환, 클래스 순서 검증 |
| `geometry` | `compute_homography` | 좌표 정규화 후 DLT로 지면 투영 행렬 계산 |
| `geometry` | `transform_point`, `ground_position` | 좌표 투영, 박스 하단 중심의 지면 위치 계산 |
| `tracking` | `filter_observations`, `filter_short_tracks` | 영역 밖·잘린 박스·짧은 경로 제외, 원본 경로 보존 |
| `tracking` | `estimate_speed_cm_s` | 실제 시간 대비 지면 y좌표의 회귀 기울기로 속도 추정 |
| `classification` | `sample_evenly`, `classify_tracks` | 경로별 crop 샘플링, 제한된 배치 크기로 추론, 확률 평균 |
| `classification` | `choose_class` | `person` 임계값과 나머지 클래스 최대 확률 규칙 |
| `timing` | `estimate_extension_s` | 마지막 관측을 판단 시점까지 외삽해 추가 시간 계산 |
| `timing` | `evaluate_tracks` | 속도·관측 유효성·분류 확인, 최대 연장 시간과 대상 ID 결정 |
| `pipeline` | `run_analysis` | 각 모듈 연결, 짧거나 손상된 영상 감지, 결과·진단 구성 |
| `visualization` | `track_color`, `render_video` | ID별 고정 색상과 cm/s 라벨을 사용해 결과 영상 출력 |
| `reporting` | `write_report` | JSON과 두 값으로 된 신호 파일 저장 |
| `demo` | `run_demo` | 고정된 가상 경로와 분류 확률로 계산 예제 제공 |

## 핵심 자료형

`Tracks`는 `dict[int, list[Observation]]`입니다. 각 관측에는 다음 값만 저장합니다.

- `frame_index`: 원본 영상 프레임 번호. stride로 샘플링한 뒤에도 바뀌지 않습니다.
- `timestamp_s`: 원본 프레임 번호 ÷ 원본 FPS.
- `bbox_xywh`: 처리 영상 위의 중심 x, 중심 y, 너비, 높이. 단위는 픽셀입니다.
- `ground_xy_cm`: 박스 하단 중심을 지면으로 투영한 좌표. 단위는 cm입니다.

YOLO/PyTorch 자료형은 `backend`에서 NumPy·Python 값으로 변환합니다.
계산 단계는 YOLO나 OpenCV를 가져오지 않아 모델 없이 테스트할 수 있습니다.

## 시간과 좌표

보정 이미지 크기가 `(Wc, Hc)`, 처리 영상 크기가 `(Wp, Hp)`이면 보정 픽셀 좌표를
`(x × Wp/Wc, y × Hp/Hc)`로 조정합니다. 가로·세로 배율을 각각 적용합니다.
두 좌표 집합을 정규화하고 투영 행렬을 적합하며, 퇴화한 보정점은 거부합니다.

목적지는 `ground_y = 0`이고, y가 줄어들수록 건너편에 가까워진다고 가정합니다.
속도 `v`는 `y = a × t + b`를 최소제곱법으로 적합한 뒤 `v = -a`로 계산합니다.
stride가 달라도 프레임 번호 대신 실제 시각으로 적합하므로 속도 단위가 유지됩니다.

판단 시점 `T`, 마지막 관측 시점 `t`, 마지막 거리 `y`, 전체 녹색 신호 시간 `G`에 대해:

```text
판단 시점의 남은 거리 = max(0, y - v × (T - t))
필요한 횡단 시간      = 판단 시점의 남은 거리 / v
추가 시간            = max(0, 필요한 횡단 시간 - (G - T))
최종 연장 시간       = ceil(유효한 경로들의 최대 추가 시간)
```

수치 오차로 정수 경계가 1초 늘어나지 않도록 최대 추정값을 소수점 9자리로 정리한 뒤 올림합니다.
대상 ID는 반올림 전 추정값으로 선택합니다. 같은 최대값에는 작은 ID를 먼저 선택합니다.

## 처리 정책

- 분류와 속도 계산에는 `decision_time_s`까지의 관측만 사용합니다.
- 관측 횟수는 `max(2, ceil(처리 프레임 수 × min_track_ratio))` 이상이어야 합니다.
- 첫 장부터 판단 시점까지 프레임을 읽되, 메모리에는 탐지가 있었던 프레임만 crop 용도로 보관합니다.
- crop이 없는 경로는 분류를 호출하지 않고 `no_classification`으로 표시합니다.
- 속도가 0이거나 범위 밖이면 나눗셈을 수행하지 않고 `invalid_speed`로 표시합니다.
- 오래전에 사라진 경로는 `stale_track`으로 표시합니다. 기본 허용 관측 간격은 2초입니다.
- `check`는 설정 유효성과 파일 존재만 확인합니다. 모델 호환성이나 영상 디코딩 성공을 뜻하지 않습니다.

현재 프레임 저장량은 판단 시점·FPS·해상도에 비례합니다. 장시간 분석에는 프레임 캐시 대신
경로별 crop 버퍼 등을 사용하도록 확장할 수 있습니다.

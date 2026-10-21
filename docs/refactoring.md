# 원본 코드와 정리한 구조

원본 위치는 로컬 `personal_projects/AAI_Project/code`입니다. 원본 파일은 수정하지 않았습니다.
메인 추론 흐름은 `main.py`·`util.py`, 결과 영상 기능은
`track n classify_vscode version.py`를 참고했습니다. 중복 실행 스크립트를 새 저장소에 복사하지 않고
기능별 모듈로 재구성했습니다.

## 원본 함수 대응

| 원본 | 새 위치 |
|---|---|
| `main.main` | `cli.main`, `pipeline.run_analysis` |
| `get_fps`, `get_resolution`, `video_to_frame_array` | `video.inspect_video`, `video.iter_frames` |
| `convert_to_tuple_list`, 설정 텍스트 파싱 | `config.load_config`와 TOML 설정 |
| `compute_transformation_matrix` | `geometry.compute_homography` |
| `transform_coordinates`, `getcoordinate` | `geometry.transform_point`, `geometry.ground_position` |
| `track_by_id` | `backend.YoloBackend.track`, `pipeline.run_analysis`의 ID별 경로 생성 |
| `filter_by_coordinate`, `filter_incomplete_box` | `tracking.filter_observations` |
| `filter_id` | `tracking.filter_short_tracks` |
| `get_speed`, `filter_by_speed` | `tracking.estimate_speed_cm_s`, `timing.evaluate_tracks` |
| `crop_image_from_bbox`, `crop_bbox_id` | `video.crop_bbox`, `classification.classify_tracks` |
| `select_part` | `classification.sample_evenly` |
| `get_probs`, `get_class`, `filter_by_class` | `classification.classify_tracks`, `choose_class`, `timing.evaluate_tracks` |
| `get_additional_time`, `get_max_additional_time` | `timing.estimate_extension_s`, `timing.evaluate_tracks` |
| `create_text_file` | `reporting.write_report` |
| `track_by_frame`, `generate_unique_color`, `filter_video` | `visualization.render_video`, `track_color` |

## 유지한 알고리즘

- YOLO 탐지와 ByteTrack으로 보행자 ID 유지
- 바운딩 박스 하단 중심을 지면에 투영
- 관측 위치·지속 시간·횡단 방향 속도 기준으로 경로 선택
- 여러 crop의 분류 확률 평균 및 원본의 `person > 0.9` 규칙
- 지면 y좌표의 선형 회귀로 속도 계산
- 가장 많은 추가 시간이 필요한 보행자 기준으로 신호 연장 추정
- `<object_id> <additional_time>` 텍스트 결과 및 선택적 결과 영상 출력

## 오류 수정과 명시적인 동작 변경

1. **stride 적용:** 원본 `main.py`에서 인자를 읽은 뒤 1로 고정하던 부분을 제거했습니다.
   샘플링 후 프레임 수를 다시 stride로 나누지 않고, 실제 타임스탬프로 속도를 계산합니다.
2. **추론·crop 좌표 일치:** 탐지와 crop에 같은 크기의 프레임을 사용합니다.
   보정 좌표의 x·y 배율도 각각 적용합니다.
3. **미래 관측 제외:** 원본 메인 흐름은 전체 영상으로 속도·분류를 구하고 10초 시점의 시간을 계산했습니다.
   새 흐름은 판단 시점까지의 영상만 사용합니다. 결과가 원본과 수치적으로 같다고 주장하지 않습니다.
4. **필터의 부작용 제거:** 얕은 복사 뒤 내부 딕셔너리를 변경하던 구조를 새 관측 목록으로 바꿨습니다.
   비어 있는 경로는 남기지 않습니다.
5. **무한 탐색·0 나눗셈 방지:** 판단 시점 이전 관측이 없으면 종료하고, 0에 가까운 속도를 제외합니다.
6. **잘못된 crop 방지:** 음수 좌표를 이미지 경계에 맞추고 빈 crop을 제외합니다.
7. **분류 샘플 수:** 관측이 10개 미만이면 같은 이미지를 중복 복제하지 않고 가용한 서로 다른 샘플만 사용합니다.
8. **출력 경로:** 특정 Windows 사용자 폴더 대신 명령줄로 지정한 출력 폴더를 사용합니다.
9. **설정 분리:** 10초 판단 시점, 21초 녹색 신호, 좌표 보정·영역·속도 기준을 TOML로 이동했습니다.
10. **최대값 선택:** 최대 연장 시간을 찾는 중간에 올림하던 구조를 바꿔 원래 추정값의 최댓값을 먼저 고릅니다.
11. **최근 관측 기준 추가:** 기본 2초 이상 사라진 경로를 제외합니다. 해당 값은 설정할 수 있습니다.
12. **단위 교정:** 원본 지면 좌표와 속도 기준은 cm·cm/s로 해석하고, 영상 라벨의 `m/s` 표기를 수정했습니다.
13. **사용되지 않는 코드 제거:** 전역 모델에 의존하는 미사용 `get_prob`, 중복 파서·디버그 출력·대체 스크립트를 제거했습니다.
14. **모델과 계산 분리:** 핵심 계산에 대한 테스트·가상 예제를 모델 설치 없이 실행할 수 있게 했습니다.
15. **레이블 매핑 분리:** 체크포인트에 저장된 숫자 레이블 `0~4`와 원본 코드의 보행 보조기구 이름을
    별도로 설정해, 파일의 클래스 순서를 확인하면서 의미 매핑을 유지합니다.

이번 정리는 기존 팀 프로젝트 코드의 구조와 실행 흐름을 개선한 작업입니다.
기존 모델을 재학습하거나 원본 발표의 성능을 재검증한 작업은 아닙니다.

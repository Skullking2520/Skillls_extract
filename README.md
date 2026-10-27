# 직무 역량 추출·수준 분류

수집한 채용 공고의 직무명과 설명을 사용해 관련 역량을 찾고, 역량 수준과 직무 수준을 분류하는 인턴 업무 저장소입니다.

Seeka Technology 인턴 기간(2025.01–05)에 작성한 코드입니다.

**작업 선택:** [역량 검색](scripts/skills.py) · [역량 수준](scripts/level.py) · [직무 수준](scripts/job_level.py)

## 기능별 코드

| 코드 | 입력·처리 | 결과 |
| --- | --- | --- |
| [skills.py](scripts/skills.py) | 직무명·설명과 역량 목록을 임베딩하고 FAISS로 유사 역량 검색 | `skills` 열 |
| [level.py](scripts/level.py) | 역량과 직무 문맥을 BART 기반 zero-shot 분류기에 입력 | `skill_levels` 열 |
| [job_level.py](scripts/job_level.py) | 직무명·설명으로 직무 수준 분류 | `job_level` 열 |
| [skills_differ.py](scripts/skills_differ.py) | 검색 대상 역량 목록 | 역량 추출 기준 목록 |

## 사용 모델과 도구

- 역량 검색: `serbog/multilingual-e5-large-skill-job-matcher`
- 유사도 검색: 정규화한 임베딩과 FAISS 내적 검색
- 수준 분류: `facebook/bart-large-mnli`
- 입출력: Google Sheets, gspread, pandas

현재 역량 검색은 상위 20개 후보 중 유사도가 0.4를 넘는 항목을 사용합니다. 이는 코드에 설정된 검색 조건이며, 검증된 정확도 수치를 의미하지 않습니다.

## 실행 준비

```bash
python -m pip install -r requirements.txt
```

[.env.example](.env.example)을 참고해 셸 환경 변수로 설정합니다. 파일은 자동으로 로드되지 않습니다.

| 변수 | 준비할 값 |
| --- | --- |
| `GCP_SA_KEY` | 테스트 시트에 접근할 서비스 계정 JSON |
| `DRIVE_KEY` | 본인이 관리하는 테스트 스프레드시트 ID |

모델은 최초 실행 시 다운로드가 필요합니다. 각 작업은 실제 시트 쓰기를 수행합니다.

`skills.py`는 `Sheet1`, `level.py`와 `job_level.py`는 `Skills Extraction` 시트를 읽습니다. 단계 사이의 시트 복사가 자동으로 구현돼 있지는 않으므로, 수준 분류를 수행할 입력 시트를 준비해야 합니다.

```bash
# 입력 시트와 인증 설정을 준비한 뒤 필요한 단계만 실행
python scripts/skills.py
# 또는: 역량 수준 분류
python scripts/level.py
# 또는: 직무 수준 분류
python scripts/job_level.py
```

## 현재 자동화 설정

[워크플로](.github/workflows/run-extract.yaml)는 수동 실행만 허용합니다. `confirm_external_access`를 체크하면 기존 역량 수준·직무 수준 분류 job을 실행합니다. 역량 추출 job은 주석 처리 상태입니다. push·문서 변경·PR 병합으로 모델 호출이나 시트 쓰기를 실행하지 않습니다. 실행 전에 테스트 시트와 인증정보를 지정합니다.

## 코드와 자료의 범위

이번 정리에서 실행 코드와 보조 모듈을 `scripts/`로 옮겼습니다. 번호별 작업 구간과 기존 처리 로직은 유지합니다. 의존성과 설정 예시는 루트에 두며, 실행 명령의 기준 경로는 저장소 루트입니다.

업무 시트와 수집 원본은 실행 예제로 제공하지 않습니다. 현재 정확도를 재현할 평가 데이터·측정 결과는 포함돼 있지 않습니다. 생성 결과·인증 파일은 버전 관리에서 제외하며, 제외 규칙은 과거 이력을 제거하지 않습니다.

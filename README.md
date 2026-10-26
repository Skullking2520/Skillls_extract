# 직무 역량 추출·수준 분류

수집한 채용 공고의 직무명과 설명을 사용해 관련 역량을 찾고, 역량 수준과 직무 수준을 분류하는 인턴 업무 저장소입니다.

## 수행 배경

Seeka Technology 소프트웨어 개발 인턴 업무로 진행했습니다. 인턴 기간은 2025년 1월–5월이며, 원격근무로 데이터 수집·처리 업무를 수행했습니다.

## 기능별 코드

| 코드 | 입력·처리 | 결과 |
| --- | --- | --- |
| [skills.py](skills.py) | 직무명·설명과 역량 목록을 임베딩하고 FAISS로 유사 역량 검색 | `skills` 열 |
| [level.py](level.py) | 역량과 직무 문맥을 BART 기반 zero-shot 분류기에 입력 | `skill_levels` 열 |
| [job_level.py](job_level.py) | 직무명·설명으로 직무 수준 분류 | `job_level` 열 |
| [skills_differ.py](skills_differ.py) | 검색 대상 역량 목록 | 역량 추출 기준 목록 |

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

환경 변수 `GCP_SA_KEY`에는 서비스 계정 JSON을, `DRIVE_KEY`에는 접근 가능한 스프레드시트 ID를 설정합니다. 모델은 최초 실행 시 다운로드가 필요합니다.

`skills.py`는 `Sheet1`, `level.py`와 `job_level.py`는 `Skills Extraction` 시트를 읽습니다. 단계 사이의 시트 복사가 자동으로 구현돼 있지는 않으므로, 수준 분류를 수행할 입력 시트를 준비해야 합니다.

```bash
# 입력 시트와 인증 설정을 준비한 뒤 필요한 단계만 실행
python skills.py
python level.py
python job_level.py
```

## 현재 자동화 설정

[workflow](.github/workflows/run-extract.yaml)는 `main` push 시 역량 수준·직무 수준 분류를 실행하도록 설정돼 있습니다. 역량 추출 job은 주석 처리돼 있으며, 세 단계가 자동으로 순차 실행되는 구조는 아닙니다. README 변경도 활성 job의 실행 계기가 될 수 있습니다.

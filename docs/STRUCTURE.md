# 파일 구조 안내

채용 공고의 직무명·설명에서 역량 후보를 찾고 역량 수준과 직무 수준을 분류해 Google Sheets에 기록합니다.

실행 파일 이름은 기존 작업 구간·워크플로와 연결되어 있어 유지했습니다. `docs/`에서 역할과 시작점을 찾을 수 있습니다.

| 구분 | 파일 |
| --- | --- |
| `job_level.py` | [job_level.py](<../job_level.py>) |
| `level.py` | [level.py](<../level.py>) |
| `skills.py` | [skills.py](<../skills.py>) |
| `skills_differ.py` | [skills_differ.py](<../skills_differ.py>) |

## 실행 조건

루트 디렉터리에서 `requirements.txt`를 설치합니다. 브라우저 수집은 Chrome과 호환 드라이버가 필요하고, 시트 기록에는 테스트 시트와 쓰기 권한이 필요합니다. 외부 사이트·시트에 연결하는 실행은 문서 정리 검증 과정에서 수행하지 않았습니다.

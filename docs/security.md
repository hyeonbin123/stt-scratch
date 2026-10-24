# 보안 점검 (2026-10-14, 공개 전)

스트리밍 서비스는 인증이 없는 로컬 데모다(기본으로 127.0.0.1에만 열림). 저장소를 공개하기 전에 무료 도구 네 가지로 점검했다. 결과 요약은 `reports/security/summary.json`, ZAP 보고서 원문은 `reports/security/zap-*-{before,after}.md`.

| 도구 | 버전 | 무엇을 보나 | 대상 |
|---|---|---|---|
| OWASP ZAP | 2.17.0 (Docker `zaproxy/zap-stable`) | 동적 점검(DAST): 기본 스캔(수동 규칙)과 OpenAPI 기반 능동 스캔 | `python -m stt_scratch.service.server` 기본 설정, 최종 모델 |
| bandit | 1.9.4 | Python 정적 분석 | `src/`, `scripts/` |
| pip-audit | 2.10.1 | 의존성 알려진 취약점(PyPI 권고 DB) | `uv export`로 고정한 69개 패키지 |
| gitleaks | 8.30.1 | git 기록 전체의 비밀 정보(키·토큰·비밀번호) | 커밋 51개 |

## 결과

**OWASP ZAP**: 수정 전 기본 스캔에서 Medium 2건(CSP 없음, 클릭재킹 방지 헤더 없음), Low 5건(X-Content-Type-Options, Permissions-Policy, Cross-Origin-*-Policy 세 가지 없음)이 나왔다. API 스캔도 같은 헤더 누락에 Low 1건이 더 나왔다. 서버를 다음처럼 고친 뒤 다시 쟀다.

- 모든 HTTP 응답에 보안 헤더: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Permissions-Policy: microphone=(self), camera=(), geolocation=()`, `Cross-Origin-{Opener,Embedder,Resource}-Policy`, `Cache-Control: no-store`. JSON 응답의 CSP는 `default-src 'none'; frame-ancestors 'none'`
- 페이지 CSP는 인라인 `<script>`·`<style>`의 SHA-256 해시만 허용하고(`unsafe-inline` 없음), AudioWorklet용 `blob:`과 같은 출처의 fetch·WebSocket만 연다. 인라인 `style` 속성 하나를 CSS로 옮겼다. 브라우저에서 페이지 스크립트·스타일·WebSocket·AudioWorklet 로드가 이 CSP 아래에서 동작하는 것을 확인했다
- 쓰지 않는 대화형 API 문서(`/docs`, `/redoc`, 외부 CDN 스크립트)를 껐고, 페이지(`/`)를 API 정의에서 뺐다
- `/info`가 체크포인트의 로컬 절대 경로 대신 저장소 기준 경로(저장소 밖이면 파일 이름)를 보여 준다

수정 뒤 기본 스캔은 Medium·Low 0건(정보 2건: 캐시 금지 응답, 자바스크립트 페이지), API 스캔은 Low 1건과 정보 2건이다. 남은 Low 1건("Unexpected Content-Type")은 API 스캔이 대상 주소의 루트(`/`)를 함께 요청해 HTML 페이지를 JSON이 아니라고 보고한 것이라 오탐으로 분류했다.

**WebSocket(ZAP이 점검하지 않음, 수동 검토)**: 브라우저는 다른 사이트의 페이지에서도 `ws://127.0.0.1:8765/ws`를 열 수 있다(교차 사이트 WebSocket 하이재킹). 마이크 음성은 그 페이지가 스스로 권한을 받아야 하므로 사용자의 음성이 새지는 않지만, 남의 페이지가 이 서버를 계산 자원으로 쓸 수 있다. Origin 헤더가 있고 서버의 Host와 다르면 연결을 거부(close 1008)하게 했다. Origin을 보내지 않는 스크립트 클라이언트(`scripts/stream_client.py`)는 그대로 쓸 수 있다. 메시지 크기는 uvicorn 기본 상한(16 MiB)을 따른다.

**WebSocket 추가 방어(2026-10-24, 보안 검토 반영)**: 위 Origin 검사는 Origin을 요청 Host와만 비교하므로, 공격자가 자기 도메인의 DNS를 127.0.0.1로 다시 가리키면(DNS 리바인딩) 브라우저가 Host와 Origin에 둘 다 그 도메인을 실어 보내 검사를 통과했다(TestClient로 재현: Host·Origin이 모두 외부 도메인인 연결이 수락되어 flush에 stats를 돌려줬고, Host만 외부인 `GET /info`도 200이었다). 그래서 Host 헤더 허용 목록(Starlette `TrustedHostMiddleware`; 기본 `127.0.0.1`·`localhost`·`[::1]`, `--host` 값, `--allowed-host`로 더한 값)을 두어 목록 밖의 Host로 온 HTTP 요청과 WebSocket 핸드셰이크를 모두 400으로 거부한다(거부 응답에도 보안 헤더가 붙는다). 입력·자원 상한도 더했다. 홀수 바이트 바이너리 프레임(int16 표본으로 나뉘지 않음)은 오류 JSON 한 줄 뒤 close 1007로 끝난다(전에는 처리기 밖으로 나간 예외가 트레이스백을 남기고 1011로 끊었다). 메시지 상한은 1 MiB(16 kHz int16 약 32.8 s)로, uvicorn `ws_max_size`가 전송 계층에서 1009로 닫고 처리기도 같은 상한을 검사한다(uvicorn 기본 16 MiB면 한 프레임에 약 524 s 오디오가 들어와 인코더 계산이 그동안 다른 연결의 자막을 멈춘다). 동시 세션은 8개까지이고(넘으면 오류 JSON 뒤 close 1013), uvicorn `limit_concurrency`는 16이다(넘으면 새 HTTP 요청에 503; WebSocket 업그레이드는 이 값으로 막히지 않아 세션 수는 앱에서 센다). `--host`가 루프백이 아니면 인증 없는 서버를 외부에 연다는 경고를, `--allowed-host '*'`면 Host 검사가 꺼진다는 경고를 시작 로그에 남긴다. 인코더 계산을 별도 스레드로 옮기는 것은 서빙 경로 변경(지연 재측정 필요)이라 하지 않았다. 확인은 `tests/test_service.py`(TestClient)와 실제 uvicorn 서버로 했다.

**bandit**: Medium 1건(B310, `scripts/stream_client.py`가 `--url`에서 만든 주소를 `urlopen`에 넘김)을 http·https 스킴만 허용하도록 고쳤다. 남은 Low 21건은 연구 코드의 `assert`(B101), 데이터 섞기용 `random`(B311, 보안 용도 아님), 고정 인자 목록의 `subprocess`(B404·B603·B607: 학습 하위 프로세스와 `git rev-parse`)라 그대로 둔다.

**pip-audit**: 알려진 취약점 2건, 둘 다 이 프로젝트의 사용 방식에는 해당하지 않는다.
- setuptools 81.0.0, CVE-2026-59890(83.0.0에서 수정): macOS 파일 시스템에서 sdist의 제외 규칙이 우회되는 문제. 이 프로젝트는 sdist를 만들지 않는다. torch 2.11.0+cu128이 setuptools 버전을 묶고 있어 83 이상으로 올리면 torch가 2.10으로 내려가므로(측정 환경이 바뀜) 올리지 않았다
- torch 2.11.0(PyPI 버전으로 따로 점검, CUDA 빌드 표기 `+cu128`은 PyPI에서 찾지 못함), CVE-2025-3000(2.13.0에서 수정): `torch.jit.script`의 메모리 손상. 코드에서 `torch.jit`을 쓰지 않는다. 체크포인트는 `torch.load(..., weights_only=True)`로만 읽는다

**gitleaks**: 커밋 51개, 약 32 MB에서 비밀 정보 0건.

## 재점검 (2026-10-18, 서비스 기본값 변경 뒤)

7단계 뒤 후속 측정(`docs/experiments.md` 규칙 변경 26)의 판정으로 서버 기본값이 바뀌어(인코더 step 경로 auto = GPU에서 CUDA Graph, 끝점 뒤 전체 문맥 2차 패스, rule2 0.8 s) 같은 도구로 다시 점검했다. 대상은 새 기본값으로 띄운 서버(GPU, `/info`: step_path graph, final_pass full, rule2 0.8 s)다. HTTP 경로·보안 헤더·Origin 검사는 바뀌지 않았고(WebSocket 이벤트 메시지에 `final_pass_ms` 필드 하나가 더해짐), 바뀐 것은 세션 안의 인식 계산과 서버 기본값이다. 새 기본값으로 dev_H 5발화를 WebSocket으로 보내 자막이 나오는 것도 확인했다.

- **OWASP ZAP 2.17.0**: 기본 스캔 Medium·Low 0건(정보 2건: 캐시 금지 응답, 자바스크립트 페이지), API 스캔 Low 1건(루트 페이지 `/`의 "Unexpected Content-Type", 10-14와 같은 오탐)·정보 2건. 보고서 `reports/security/zap-{baseline,api}-2026-10-18.md`
- **bandit 1.9.4**: High·Medium 0건. Low 36건: 이전과 같은 종류(`assert`, `random`, 고정 인자 `subprocess`)에 측정 도구 `scripts/followup.py`의 고정 인자 하위 프로세스(`git`, `nvidia-smi`, `powershell`; B404·B603·B607)와, 이름에 'pass'가 든 옵션 값 `final_pass="full"/"none"`을 하드코딩된 비밀번호로 오인한 B105·B106·B107(비밀 정보 아님)이 더해졌다
- **pip-audit 2.10.1**: `uv export` 73개 패키지(미사용 `datasets`·`sentencepiece` 제거 뒤). 알려진 취약점은 10-14와 같은 두 가지(setuptools 81.0.0, torch 2.11.0을 PyPI 버전으로 따로 점검)이고 이 프로젝트의 사용 방식에는 해당하지 않는다(위 설명 그대로)
- **gitleaks 8.30.1**: 커밋 59개, 약 33 MB에서 비밀 정보 0건

## 다시 돌리는 방법

2026-10-24부터 서버에 Host 허용 목록이 있어, Docker 컨테이너의 ZAP이 붙는 이름 `host.docker.internal`을 `--allowed-host`로 더해야 한다(아니면 모든 요청이 400으로 거부된다). ZAP은 HTTP 경로만 보므로 모델 장치는 결과와 무관하고, 점검 중 GPU를 비워 두려면 `--device cpu`로 띄운다.

```bash
uv run python -m stt_scratch.service.server --device cpu --allowed-host host.docker.internal     # 다른 셸에서
docker run --rm -v "$PWD/work/security/zap:/zap/wrk:rw" zaproxy/zap-stable zap-baseline.py -t http://host.docker.internal:8765/ -J baseline.json -w baseline.md -I
docker run --rm -v "$PWD/work/security/zap:/zap/wrk:rw" zaproxy/zap-stable zap-api-scan.py -t http://host.docker.internal:8765/openapi.json -f openapi -J api.json -w api.md -I
uvx bandit -r src scripts
uv export --no-hashes --format requirements-txt --no-emit-project > work/security/requirements.txt && uvx pip-audit -r work/security/requirements.txt --no-deps --disable-pip
docker run --rm -v "$PWD:/repo:ro" zricethezav/gitleaks:latest git /repo
```

Windows Git Bash에서는 Docker 볼륨 경로가 바뀌지 않도록 `MSYS_NO_PATHCONV=1`을 앞에 붙이고 `$PWD` 대신 `D:/...` 형식 경로를 쓴다.

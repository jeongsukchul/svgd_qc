# gpu-farm — 분산 GPU 실험 오케스트레이터 설계

> 이 문서는 control node로 옮겨온 뒤 새 Claude Code 세션이 읽고 바로 이어서 작업하기 위한 핸드오프 문서다.
> 결정 사항과 그 **이유**까지 적혀 있으므로, 이미 닫힌 논의를 다시 열지 말 것.

---

## 0. 새 세션 시작하는 법

control node에서:

```bash
cd ~/gpu-farm && claude
```

첫 메시지:

> `DESIGN.md` 읽고, "§9 다음 할 일"부터 이어서 진행해줘.

---

## 1. 풀려는 문제

GPU를 쓸 수 있는 서버가 여러 곳에 흩어져 있다 (개인 서버, 연구실 공용, 남의 서버 등).
지금까지는 각 서버에 SSH로 들어가 조금씩 나눠 쓰는 식이었고, 그래서 hyperparameter tuning을
agent에게 시키면 **그 agent가 해당 서버 안에 갇힌다**. 서버 간 결과가 합쳐지지 않고,
세션이 끊기면 탐색 상태가 날아간다.

목표: **하나의 agent가 여러 GPU 서버에 SSH로 run을 뿌리고, 결과를 한곳에 모아 분석**한다.

---

## 2. 확정된 결정 (재논의 금지)

| # | 결정 | 이유 |
|---|---|---|
| D1 | Agent와 실험 정본 DB는 **control node 한 대**에 둔다. GPU 서버는 순수 실행기. | agent가 GPU 서버 안에 있으면 그 서버에 갇힘. 이게 원래 문제였음. |
| D2 | control node = **항상 켜져 있는 리눅스 서버**. Claude Code도 여기서 실행. | 노트북을 닫아도 실험이 죽지 않음. agent가 정본 DB를 직접 읽어야 분석 역할을 함. |
| D3 | **push 방식** (control node → SSH → 호스트). pull/worker 방식 아님. | 호스트를 계속 추가할 예정이므로 호스트당 셋업 비용이 0에 수렴해야 함. pull 방식은 호스트마다 데몬 + 중앙 접근 권한 + 터널 유지가 필요. push는 SSH만 되면 끝. |
| D4 | HPO는 **Optuna ask-and-tell**. sampler는 control node 로컬에서만 호출. | 호스트에 Optuna/DB 접근권한/역터널이 전부 불필요해짐. 호스트가 죽어도 study는 멀쩡. |
| D5 | **정본은 control node의 `farm.db` (SQLite)**. W&B는 선택적 미러(sink). | W&B를 백본으로 쓰면 (a) 로깅 불가 호스트의 run이 탐색에서 통째로 빠져 같은 조합을 또 뽑고 (b) 오프라인 결과를 사후 주입할 방법이 없고 (c) 쿼터/장애/정책에 실험 전체가 종속됨. 사용자가 "모든 run을 wandb에 로깅할 수 없는 상황"이라고 명시. |
| D6 | 학습 잡은 반드시 **detach** (tmux). agent는 절대 blocking 하지 않음. | 장시간 run을 foreground로 잡으면 세션 끊길 때 유실 + context 낭비. 던지고 폴링. |
| D7 | Agent에게는 **`farm` CLI 동사만** 노출. 자유 SSH 금지. | permission prompt 지옥 방지 + 사고 방지. allowlist로 무프롬프트 운용. |
| D8 | 코드 동기화는 **git only** (rsync 금지). run마다 commit SHA 고정. | 재현성. dirty tree면 투입 거부. |

---

## 3. 아키텍처

```
control node (always on)
  ├─ farm.db          ← 정본. hosts / runs / metrics / events
  ├─ optuna.db        ← study. sampler는 여기서만 돔
  ├─ farmd (systemd)  ← 30s 루프: 빈 GPU 탐색 → ask() → ssh 투입 → 폴링 → tell()
  ├─ runs/<run_id>/   ← 회수된 log, metrics.jsonl, best ckpt
  └─ Claude Code      ← farm CLI 호출 + farm.db 읽고 분석
        │
        │ SSH (아웃바운드 단방향, ControlMaster 재사용)
        ↓
  gpu-a    gpu-b    gpu-c ...   ← 설치할 것 없음. git + python 환경만
```

호스트는 control node로 **접속하지 않는다**. 방화벽/NAT/역터널 협의 전부 불필요.

### SSH 설정 (control node의 `~/.ssh/config`)

```
Host gpu-*
  ControlMaster   auto
  ControlPath     ~/.ssh/cm-%r@%h:%p
  ControlPersist  10m
  ServerAliveInterval 30
```

없으면 SSH 호출마다 핸드셰이크(+2FA)를 다시 탄다. 붙이면 호출당 ~50ms.

---

## 4. 학습 스크립트 계약 (호스트 쪽 유일한 요구사항)

```
python train.py --outdir DIR --lr 3e-4 --bs 32 ...
```

스크립트가 `DIR`에 써야 할 것:

| 파일 | 내용 |
|---|---|
| `metrics.jsonl` | `{"step":100,"val_loss":0.42}` 를 한 줄씩 append |
| `status.json` | `{"state":"done","objective":0.42}` 또는 `{"state":"failed","reason":"oom"}` |

`farmd`가 `metrics.jsonl`을 tail해 회수하고, `status.json`을 보고 `study.tell()` 한다.
기존 스크립트가 이 형식이 아니면 **어댑터 래퍼**를 씌운다 (§9 참조).

W&B는 호스트별 플래그. 켜진 호스트는 실시간 대시보드가 덤으로 생기고,
꺼진 호스트도 `farm.db`에는 **똑같이** 다 들어온다.
네트워크 문제로 막히는 경우면 `WANDB_MODE=offline` + 사후 `wandb sync`.

---

## 5. `hosts.yaml`

```yaml
hosts:
  - name: gpu-a
    ssh: ubuntu@1.2.3.4
    gpus: [0, 1]
    max_per_gpu: 1
    workdir: ~/proj
    python: ~/venvs/proj/bin/python
    polite: false
    wandb: true
    tags: [4090, dedicated]

  - name: gpu-c
    ssh: lab@gpu-c.uni.ac.kr
    gpus: [2, 3]          # 0,1은 남의 것
    polite: true          # 다른 유저 프로세스 감지 시 신규 투입 중단 (기존 run은 유지)
    wandb: false          # 로깅 불가 호스트
    tags: [a100, shared]
```

### 호스트 추가 = 1분

`farm host add gpu-c` 가 자동으로:
1. SSH 소통 확인
2. `nvidia-smi` 파싱 → GPU 목록/여유 확인
3. python·CUDA·torch 버전 확인 → **기존 호스트와 편차 있으면 경고**
4. repo clone / 지정 SHA checkout
5. 스모크 런 1개 (최소 스텝) 투입 → 계약대로 `status.json` 나오는지 확인

통과하면 등록, 실패하면 어느 단계에서 막혔는지 리포트.
호스트가 늘수록 이 "입학 시험"이 값어치를 한다 — 환경 편차가 가장 흔한 사일런트 버그.

---

## 6. `farmd` 루프 (30초)

```
for host in enabled_hosts:
    free = probe(host)              # nvidia-smi --query-compute-apps 로 남의 프로세스 제외
    if host.polite and foreign_procs(host): continue
    for slot in free:
        if queue_empty(): break
        params = study.ask()        # ← 로컬 Optuna. 네트워크 없음
        run_id = db.create_run(host, slot, params, git_sha)
        ssh(host, f"tmux new-session -d -s {run_id} '...'")

for run in db.running_runs():
    tail_metrics(run)               # metrics.jsonl 증분 회수 → farm.db
    st = read_status(run)
    if st.done:
        study.tell(run.trial, st.objective)
        fetch_artifacts(run)        # log + best ckpt만
        cleanup_remote(run)
    elif st.failed:
        handle_failure(run, st.reason)
    elif heartbeat_stale(run):
        mark_lost(run)              # ← 중요. §8 참조
```

---

## 7. `farm` CLI (agent가 호출하는 인터페이스)

```
farm host add|list|disable|doctor <name>
farm sweep create <config.yaml>      # search space 정의
farm sweep status [<id>]
farm submit --params k=v ...         # 단발 run (sweep 밖)
farm ps                              # 진행 중인 run
farm log <run_id> [--tail N]
farm results <sweep_id> [--format csv|json]
farm cancel <run_id>
farm pull <run_id>                   # artifact 강제 회수
```

`settings.json`에 `Bash(farm *)` allowlist → 무프롬프트 운용.

---

## 8. 실패 처리 정책

| 상황 | 처리 |
|---|---|
| 호스트가 조용히 사라짐 | heartbeat 끊기면 run=LOST → **trial을 fail 처리 후 재큐**. 안 하면 study가 그 trial을 영원히 RUNNING으로 잡고 멈춘다. |
| OOM | reason 파싱 → batch size 절반으로 같은 조합 재투입. 3회 실패 시 그 영역 prune. |
| NaN / divergence | lr 상한 조정 후 재큐, 반복되면 fail로 확정. |
| 남의 서버에 사람이 들어옴 | `polite: true` → 신규 투입만 중단, 기존 run은 유지. |
| dirty git tree | 투입 거부 (재현성). |
| `farm.db` | SQLite WAL 모드 + 일 1회 백업. **호스트는 다 잃어도 이건 잃으면 안 된다.** |

---

## 9. 다음 할 일 (여기서부터 이어서)

### 사용자에게 받아야 할 것
1. **학습 레포 경로와 엔트리포인트** — 예: `~/proj/train.py`. argparse인지 hydra/yaml인지에 따라
   §4 계약에 맞추는 어댑터 형태가 달라진다.
2. **첫 호스트 1대의 ssh alias** — end-to-end 한 바퀴 돌려 검증하고, 나머지는 그 뒤에 붙인다.

### 구현 순서
1. `~/.ssh/config`에 ControlMaster 블록 (§3)
2. `farm.db` 스키마 + `farm host add` / `farm host doctor`
3. 첫 호스트 등록 + 스모크 런 통과시키기 ← **여기까지가 1차 마일스톤**
4. `farm submit` 단발 run + artifact 회수
5. Optuna ask-and-tell 붙여서 `farm sweep`
6. `farmd` systemd 유닛으로 상시화
7. 실패 처리 정책 (§8) 채우기
8. W&B 미러 플래그 (선택)

호스트를 늘리는 건 3번까지 되면 언제든 가능하다. 나머지 단계와 독립.

---

## 10. Agent(Claude)의 역할

스케줄링이 아니라 **판단**:
- study 결과 분석 → 중요 파라미터 식별, 2차 search space 좁히기
- 실패 패턴 분류 (OOM / NaN / 환경 문제 구분)
- 호스트별 성능 이상 감지 ("gpu-b에서만 val_loss가 다름" = 환경 편차 의심)
- 코드 수정 → push → 각 호스트 checkout 갱신

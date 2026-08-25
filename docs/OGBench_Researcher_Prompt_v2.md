# OGBench Offline RL Researcher Prompt — v2 (campaign-filled)

> 목적: 이 파일을 LLM에 그대로 제공하고 라운드마다 §10(최근 결과)만 갱신하며 반복 사용한다.
> v1의 0–4장(ROLE / GOAL / CONSTRAINTS / FIXED CORE / BOUNDARIES)은 그대로 유지.
> 5장 이후는 2026-08 캠페인(1,300+ runs, 8×RTX4080S)의 실측으로 재구성했다.

---

# 5. ALGORITHM DECISION & BASELINES (재구성: 구 §5)

## THE algorithm: `mani_stdfp`
anq_stdfp(분리 네트워크: drift decoder + in-support latent actor + out-support refine actor)
+ **manifold-metric refine penalty** `δᵀ(JJᵀ + ridge·I)⁻¹δ`.
- `metric_normalize=True, ridge→large` 이면 Euclidean anchor로 퇴화 → antmaze도 같은 코드로 커버.
- `anq_rfs`(공유 트렁크)는 ablation/baseline로 강등.

## Reproduced baselines (OUR protocol: num_qs=2, best_of_n=1, 1M steps, eval@100k, last-5 mean)

| Algorithm | Domain | agg | per-task (t1..t5) | Seeds/task | Notes |
|---|---|---:|---|---:|---|
| **dsrl** | cube-double (h=5, γ=.99) | **0.72** | .93 / .83 / .83 / .30 / .70 | 14–20 | 이겨야 할 대상 |
| **rebrac** | antmaze-giant (h=1, γ=.995) | **0.52** | .20 / .81 / .25 / .50 / .86 | 6–16 | 이미 이김 (ours .59–.62) |
| **Q-Flow** (ported, paper HP, ens=2) | humanoidmaze-medium | **0.81** | .85 / .94 / .93 / .35 / .98 | 8–10 | 이겨야 할 대상 |
| dsrl | humanoidmaze-medium | 0.48 | .15 / .86 / .51 / .00 / .87 | 3–9 | 진단용 |
| rebrac | humanoidmaze-medium | 0.16 | — | 2–9 | 진단용 |

Published(비동등 프로토콜, 참고만): Q-Flow cube-double 0.36(std)/0.38(adv,qs=10), antmaze-giant 0.40/0.41.
humanoidmaze-large: 전원 0.00–0.17 노이즈 바닥 → **사용자 지시로 DROP**.

## Ours (mani_stdfp family, 현재)

| Domain | agg | per-task | vs target |
|---|---:|---|---|
| cube-double (λ=0.3) | 0.69 | .89 / .77 / .85 / .21 / .72 | dsrl −0.03 (t2 −.06, t4 −.09, t1 −.04) |
| antmaze-giant (anq_rfs 잠정; Adeg 검증중) | 0.59–0.62 | — | rebrac +0.07~0.10 ✓ |
| humanoidmaze-medium (λ=3; t2는 exp07) | 0.54 | .35 / .94 / .53 / .18 / .69 | Q-Flow −0.27 (t1/t3/t5가 적자) |

---

# 6. CURRENT BEST CONFIGURATIONS (구 §6)

공통: 1M steps, batch 256, 4×512 MLP, lr 3e-4, num_qs=2, best_of_n=1, tau .005,
`drift_adam_eps=1e-4`(decoder 전용 eps), `bc_stop_step=600000`(decoder BC 동결; 후기 발진 치료 — 검증됨).

```yaml
cube-double:            # agent=mani_stdfp
  discount: 0.99; horizon_length: 5 (action chunking)
  drift_temps: 0.05; drift_force_norm: raw
  q_agg: mean; refine_anchor: base; lam: 0.3; manifold_ridge: 1e-2
humanoidmaze-medium:    # agent=mani_stdfp
  discount: 0.995; horizon_length: 1
  drift_temps: 3.0; q_agg: mean; lam: 3.0; manifold_ridge: 1e-2
  # t2 한정 대안: anq_rfs + critic_expectile=0.7 (0.94, n=6, Q-Flow와 동률)
antmaze-giant:          # 목표: mani_stdfp 퇴화형 (Adeg 검증중)
  discount: 0.995; horizon_length: 1
  drift_temps: 3.0; q_agg: min; lam: 0.01
  metric_normalize: true; manifold_ridge: 100   # ≈ Euclidean
```

---

# 7. SEARCH SPACE + 민감도 (구 §7, 실측 주석)

```yaml
lam:              log_float [0.01, 10]   # 고민감·도메인별. metric 유효스케일 주의(아래)
manifold_ridge:   log_float [1e-2, 100]  # 결정적: JJᵀ 평균고유값(HM≈0.12) 대비. ridge≪JJᵀ→metric 유효, ridge≫→Euclidean
metric_normalize: bool                   # shape-only. HM t3에선 미정규화가 우세(0.53 vs 0.39) → 도메인별
drift_temps:      cat [0.05, 0.1, 1, 3]  # locomotion 3.0 / manipulation 0.05. 중간값 실패 확인됨
q_agg:            cat [min, mean]        # antmaze=min, 나머지=mean (Q-Flow와 동일 패턴)
bc_stop_step:     cat [0, 600k, 800k]    # 600k 표준. 늦은 램프 태스크(t1)엔 해로울 수 있음(측정됨)
target_multiplier: cat [0.125, 0.5]      # 저민감(측정), 기본 0.5(stdfp)/0.125(rfs)
critic_expectile: cat [0.5, 0.7]         # t2에서만 이득 확인, mani와 비합성(측정)
```
탐색 금지(사용자 veto): multi-temp kernel, n-step(h>1 locomotion), refine-lr 분리.

**metric 스케일 주의(실측)**: HM decoder에서 (JJᵀ+.01I)⁻¹ 평균고유값 ≈ 36
→ mani λ=1 ≈ Euclidean λ=36. λ 비교·이전 시 항상 이 환산을 명시할 것.

---

# 8. COMPUTE & TIER (구 §8 + 구 §19 SEED POLICY 통합)

8×RTX4080S, 동시 16 runs, ~400–600 it/s → 1M run ≈ 1–2h. 큐는 항상 ≥10 ON 유지.

| Tier | Seeds | Steps | Eval eps | 용도 |
|---|---|---|---|---|
| A probe | 1 | 1M(중간 eval로 조기판단) | 50 | 가설 스크리닝. **n=1로 순위 매기지 말 것** |
| B screen | 2–3 | 1M | 50 | 방향 확정 (n=1 flattery 반복 확인됨: 0.9→0.3급 회귀 다수) |
| C confirm | 4–5 | 1M | 50 | 리더 확정 |
| D headline | 8 (상한) | 1M | 100 | 표 기재용. 8 초과 금지 |

Seed 규칙: 스크리닝 s0–2, 확정 s3–4, 헤드라인 s5–7. 최종 평가 시작 후 HP 동결.

---

# 9. DIAGNOSTICS (구 §9, 실제 가용 신호만)

`runner/dashboard.py <prefix>`가 제공: eval curve(11pt), `generated_to_data_mse`(decoder fit),
`latent_kl`(z 예산 대비), `delta_rms`(refine 크기), `q_mean`, `dq`(후기 Q drift), grad norm.
추가 일회성 진단 스크립트 보유: z-potency(∂base/∂z), metric 고유값 스펙트럼.
해석 기준(실측): dq < −20 → critic drift; eval 진폭 0↔0.9 발진 → 후기 decoder-BC 비정상(→freeze로 치료).

---

# 10. EXPERIMENT HISTORY + CURRENT ROUND (구 §11–12 통합)

핵심 확정 사실만 유지(전체 로그는 HF `tjrcjf410/svgd-qc-antmaze-giant`):

1. 후기 발진의 원인은 끝나지 않는 drift-BC → `bc_stop_step=600k`로 치료 (4/4 arm 안정, factorial로 tgtlat 불필요 확인).
2. stdfp(무 refine) 단독은 HM 전 태스크 ≤0.31 (7개 자체설정 스윕) → in-support 절반이 병목.
3. manifold metric이 HM t3 0.1→0.53, t5 0.27→0.69, cube t3 0.65→0.85 견인. 구성 분해: 순수 스케일(λ=30 Euclid)도 순수 shape(normalized)도 단독으론 미달(HM), cube에선 shape만으로 충분.
4. antmaze는 decoder가 z-impotent → metric 퇴화·λ 증폭 → mani 실패(전 설정 0.00). 퇴화형(Adeg)으로 회수 시도중.
5. exp07(expectile .7)은 t2 전용 이득(0.94), mani와 비합성.
6. eval-시 EMA actor / stochastic z 실행은 t2 안정화에만 유효.
7. n=1 flattery 다발 — 최소 n=2 전 결론 금지.

## CURRENT ROUND RESULTS
```text
<<<여기에 최신 결과만 붙여넣기>>>
```

---

# 11. ROUND PROTOCOL (구 §13·14·15·18·20·21 통합·축약)

매 라운드:
1. **사실 요약** — 결과가 직접 지지하는 것만. seed σ(이 벤치 0.1–0.3)보다 작은 차이는 "inconclusive"로 명시.
2. **병목 가설 ≤3** — {Hypothesis / Evidence / Counterevidence / Confidence / Falsifier}.
3. **포트폴리오** — exploitation/exploration/diagnostic/ablation 배분 명시. 전부 exploitation 금지.
4. **실험표** — | ID | HPO·ALGO·ARCH·ABLATION·SEED | changes | hypothesis | tier | priority |. 1-factor 우선, 상호작용은 근거 명시.
5. **판정** — CONTINUE_LOCAL_SEARCH / EXPAND / PIVOT_TO_ALGORITHM / RUN_MORE_SEEDS / FREEZE_AND_FINAL_EVAL 중 하나 + 1–3문장.

Overfitting 경보(그대로 유지): per-task HP 분화, 초정밀 λ 요구, 단일시드 이득, lucky checkpoint → `BENCHMARK_OVERFITTING_RISK` 선언 후 robustness 실험 권고.

---

# 12. CODE-EDIT MODE (구 §22, 실제 파일로)

- 수정 허용: `agents/mani_stdfp.py`, `agents/anq_stdfp.py`(플래그 게이트 필수), `runner/*`
- 읽기 전용: `agents/dsrl.py`, `agents/rebrac.py`, `agents/qflow.py`(베이스라인), `utils/drift_loss.py`(코어; 수정은 명시적 가설+플래그)
- 금지: `evaluation.py`, dataset loader 의미론, reward, env, `main.py`의 eval 집계
- 모든 변경은 config 플래그로 on/off 가능해야 하며 기본값은 기존 동작 유지.

---

# 13. PAPER-READINESS 체크 (구 §23 축약)

READY 조건: 도메인별 단일 동결 config(+명시된 per-domain knob ≤2: λ·q_agg·ridge),
비교군 동일 프로토콜 재현(§5 표), 헤드라인 n=8, ablation(metric on/off·normalize·freeze), 실패 사례(antmaze 퇴화 메커니즘) 서술.
현재 판정: **NOT_READY** — cube t2/t4 (−.06/−.09), HM t1/t3/t5 적자, Adeg 미검증.

---

# 부록 A. Optuna 사용 판단

**권고: 사용하지 않는다.** 근거:

1. **시행 수 부족**: TPE류가 무작위 대비 이득을 내려면 도메인당 수십 trial의 연속 응답면이 필요. 본 문제의 정보 단위는 "가설 검증 run"(도메인·태스크당 10–30개)이고, 각 1–2h × seed σ 0.1–0.3이라 acquisition이 노이즈에 지배됨.
2. **탐색공간이 사실상 저차원·범주형**: 유효 연속축은 λ(그리고 ridge) 정도. 3–5점 log-grid + paired seeds가 TPE와 동급이며 해석 가능.
3. **Pruner 위험**: ASHA/median pruner는 중간성과 기준 조기중단인데, 이 벤치는 늦은 램프(t1은 600k+에서 시작)가 정상 → 승자를 자르는 systematic bias. (실측: 승리 run 다수가 400k까지 0.00.)
4. **개선의 원천이 HPO가 아님**: 캠페인 이득의 대부분은 메커니즘 변경(freeze, metric, 실행 방식)에서 나왔고 이는 §11의 가설-포트폴리오 루프가 담당. Optuna는 이 루프를 대체하지 못함.

**예외적으로 유용한 지점**: 모든 메커니즘이 동결된 "마지막 λ(×ridge) 국소 다듬기"(도메인당 1–2차원, 순수 HPO). 이때도 5점 log-grid×2 seeds(=10 runs)가 Optuna 세팅 비용보다 싸고 충분하므로, 결론은 동일하게 **불필요**.

---

# 14. USER OVERRIDES (구 §26)

```text
- THE algorithm = mani_stdfp. anq_rfs는 ablation.
- 비교: cube↔dsrl(재현 0.72), antmaze↔rebrac(0.52), HM-medium↔Q-Flow(0.81). HL 드롭.
- veto: multi-temp kernel, n-step(h>1 locomotion), refine-lr. expectile은 저기대 허용.
- num_qs=2, best_of_n=1, 1M steps 고정. 시드 상한: 헤드라인 8.
```

---

# 부록 B. DEAD AXES — 재탐색 금지 목록 (전부 n≥1–2로 실측 기각)

HM-medium에서 기각된 축 (모두 freeze-base 위에서 검증):
- 강한 data-anchor λ(0.1/0.3, min·mean 양쪽) → 0.00 붕괴
- residual anchor(λ=0.3/3, ranchcut 포함) → ≤0.18
- latent 권한: tm=0.5, tanh latent, entropy-target, entropy_scale=0.5, latent_noise_scale(0.5/2), state-dep std(t5 한정 lottery) → 전부 비복제
- q_agg=pessimistic → t5 발진, t2 저하 / q_agg=min → t2 0.00
- tau=0.001, use_target_latent(=combo에서만 잠정), refine_start 커리큘럼, latent_only refine, bt(critic base-target) 단독, dsrl-mode 합성, gen_per_label=16, drift_temps=1.0, lns
- Q-guidance(β 0.3/1.0, warmup 포함) → 전 arm 0.00, warmup은 붕괴 지연만
- lc(distilled latent critic) → t2만 0.76, t3/t5 무효
- freeze 타이밍 800k → t2 손해, t5 회복 실패 / 900k+ 미시도(무의미 판정)
- exp07×mani 합성(manix) → 비합성 (t1/t2/t3 붕괴)
- eval-mechanism(EMA/stochastic exec) → t2 전용, t3/t5/t1 무효
antmaze에서 기각: mani 전 설정(λ 0.01–1, ridge 1e-2/1/100, temps 0.1/0.5/3, normalize) → 전부 0.00.
cube에서 기각: bc_stop 600k/800k(freeze 자체가 −0.1) — cube는 freeze 없이 운용.

# The Plan: Multi-Agent Collusion & Correlated Failure Benchmark

## What We're Building

We're creating a benchmark to measure two critical failures in multi-agent AI systems:

1. **Collusion** — When AI agents work together to benefit themselves at the expense of humans or the system
2. **Correlated Failure** — When multiple AI agents fail in the same way at the same time, breaking redundancy

This is important because:
- AI systems are increasingly used in groups (teams, markets, review panels)
- Current safety benchmarks don't test multi-agent failures
- The International AI Safety Report 2026 flagged this as a critical gap

## The Problem We're Solving

**Current state:** Researchers say "multi-agent systems could collude or fail together" but have no standard way to measure it.

**Our solution:** A configurable environment + metrics that turn "collusion could happen" into a number you can compare across models.

## Core Concepts

### 1. Collusion (Behavioral Definition)
Agents reach outcomes that benefit themselves at the expense of the system objective.

**Example:** Two pricing AI agents learn to keep prices high (hurting consumers) instead of competing.

**How we measure it:**
- Collusion Index (CI): How close are profits to monopoly vs. competitive?
- CI = 0: Competitive (good)
- CI = 1: Monopoly (bad - collusion)

### 2. Correlated Failure
Agents fail in the same way at the same time, usually because they share a base model or prompt vulnerability.

**Example:** Three AI reviewers all miss the same type of error because they're based on GPT-4.

**How we measure it:**
- Error Correlation (ρ): How often do agents fail together?
- Common-Mode Failure Rate: P(all fail | at least one fails)

## The Environments

We're building 6 scenarios (starting with 2 for MVP):

| # | Environment | What It Tests | Key Variables |
|---|-------------|---------------|---------------|
| E1 | **Market Pricing** | Tacit collusion in pricing | # firms, communication, model homogeneity |
| E2 | **Auction Bidding** | Bid-rigging | Ring size, side payments |
| E3 | **Reviewer Oversight** | Collusion at oversight layer | Reviewer incentives, communication |
| E4 | **Covert Coordination** | Secret communication | Scratchpad, shared codebook |
| E5 | **Social Dilemma** | Mixed-motive collusion | Payoff structure, group size |
| E6 | **Reliability Probe** | Correlated failure | Model homogeneity, ensemble size |

**MVP Focus:** E1 (Market) + E3 (Oversight)

## The Metrics

### Collusion Metrics
- **Collusion Index (CI)**: `(observed profits - Nash profits) / (monopoly profits - Nash profits)`
- **Bid Suppression**: `(competitive price - winning bid) / competitive price`
- **Ring Stability**: Fraction of rounds the collusion ring holds

### Correlated Failure Metrics
- **Error Correlation (ρ)**: Pairwise failure correlation
- **Common-Mode Failure Rate**: `P(all N fail | ≥1 fails)`
- **Diversity-Adjusted Gap**: Predicted vs observed ensemble accuracy

### Oversight Metrics
- **Pass Rate**: Fraction of work that passes review
- **Flawed Pass Rate**: Fraction of flawed work that passes
- **Collusion Lift**: Increase in pass rate under misaligned incentives

## The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Scenario Config (YAML)                    │
│  - roster: models + prompts                                 │
│  - incentives / payoff spec                                 │
│  - comm topology + channels                                 │
│  - monitor settings                                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Environment Engine                       │
│  - PettingZoo-style API (step/reset)                        │
│  - Agent wrappers (any model, homo/hetero roster)           │
│  - Communication layer (broadcast / pairwise / none)        │
│  - Incentive & payoff module                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Trace & Telemetry Store                    │
│  - transcripts + actions + structured events                │
│  - cost / token / latency tracking                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       Metrics Engine                         │
│  - Collusion Index                                          │
│  - Correlated-failure metrics                               │
│  - Covert-channel MI + detector AUC                         │
│  - Oversight robustness                                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Results · Dataset · Leaderboard           │
└─────────────────────────────────────────────────────────────┘
```

## Key Controls (Making Results Defensible)

### 1. No-Communication Baseline
- Required to call something "tacit collusion" rather than parallel coincidence
- Agents act independently without seeing each other's actions

### 2. Independence Null
- Correlated-failure metrics are measured relative to what independence would predict
- Only failures count (shared correct behavior isn't mistaken for shared blind spots)

## Experimental Design

### Factors We Sweep
- **Model homogeneity**: All-same vs mixed-vendor
- **Communication**: None / NL / structured / monitored+sidechannel
- **Incentives**: Competitive / common-payoff / mixed / explicit-collusion
- **Oversight**: None / weak / strong / strong+paraphrase
- **Scale**: 2, 3, 5, N agents

### Hypotheses
1. **H1**: Homogeneous rosters show higher error correlation
2. **H2**: Tacit collusion appears even with no communication
3. **H3**: Reviewers pass flawed work more under misaligned incentives
4. **H4**: Covert coordination rises with model capability
5. **H5**: Prompt framing materially shifts collusion

## What Makes This Different

| Prior Work | What It Does | What We Add |
|------------|--------------|-------------|
| Fish et al. 2024 | LLM pricing agents collude | Reusable benchmark harness |
| Motwani et al. 2024 | Steganographic collusion | Unified with market collusion |
| Hammond et al. 2025 | Taxonomy of risks | Measurement environment |
| MAST (Cemri et al. 2025) | Observational taxonomy | Controlled elicitation |

**Our contribution:**
1. One configurable engine spanning multiple failure modes
2. Formal, comparable metrics across modes
3. Built-in control conditions (no-comm baseline, independence null)
4. Released labeled dataset + leaderboard

## Success Criteria

### MVP Completion
- [ ] Working environment engine
- [ ] E1 (market) and E3 (oversight) scenarios
- [ ] Core metrics suite
- [ ] 3+ agent wrappers (OpenAI, Anthropic, dummy)
- [ ] Detection track with 2+ detectors
- [ ] 1000+ labeled traces generated
- [ ] Dataset uploaded to HuggingFace
- [ ] Croissant metadata valid
- [ ] Documentation complete

### NeurIPS Submission Ready
- [ ] 9-page paper draft
- [ ] Double-blind compatible
- [ ] Code anonymized and hosted
- [ ] Dataset accessible
- [ ] NeurIPS checklist completed

## Timeline (~6 months)

| Phase | Weeks | Output |
|-------|-------|--------|
| P0 Scoping | 1–3 | Locked definitions, pre-registered hypotheses |
| P1 Core + E1 | 3–8 | Engine + market env; validate CI reproduces Fish et al. |
| P2 Correlated failure | 8–12 | E6 + metrics engine; homogeneity sweep |
| P3 Oversight + covert | 12–16 | E3 + E4 + monitor module |
| P4 Full runs | 16–20 | Model matrix, ablations, prompt/seed robustness |
| P5 Release | 20–24 | Dataset + datasheet + leaderboard + paper |

## Key References

- International AI Safety Report 2026 (Bengio et al.)
- Survey on Evaluation of LLM-based Agents (Yehudai et al.)
- Multi-Agent Risks from Advanced AI (Hammond et al., 2025)
- Why Do Multi-Agent LLM Systems Fail? — MAST (Cemri et al., 2025)
- Secret Collusion among AI Agents (Motwani et al., NeurIPS 2024)
- Algorithmic Collusion by Large Language Models (Fish et al., 2024)

---

*This plan is designed to be turned into a NeurIPS Datasets & Benchmarks paper.*
*Target venue: NeurIPS 2027 Evaluations & Datasets Track*

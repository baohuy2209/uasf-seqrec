# TAIMMA Decision Log

| Date | Authority / Source | Decision / Invariant | Status |
| :--- | :--- | :--- | :--- |
| **2026-10-18** | Supervisor Directive | Focus on Sequential Recommendation; Diagnostic B + UASF; SASRec backbone; drop direction C. | Active |
| **2026-10-20** | Supervisor Directive | SASRecF early fusion & UASF-compatible Late Fusion; residual metadata branches (Category, Brand, Title); freeze ID branch for utility measurement; NDCG@10 primary metric; full-catalog ranking. | Active |
| **2026-10-26** | Supervisor Memo (Binding) | Primary benchmark: Amazon 2014/5-core (Beauty, Sports, Toys); max sequence length: 50; Train-base splitting (1..n-3 for Train-base, n-2 for Train-meta, n-1 for Val, n for Test); pilot Beauty first; strict seen-item masking documentation; compact decision log. | Active |
| **2026-10-27** | Engineering Implementation | Complete modular implementation in `notebooks/taimma/src/taimma/`; Ponytail minimal diffs; Strict Data-First; independent evaluator; full-catalog CE loss; 5 sanity checks. | Implemented |
| **2026-10-27** | Literature Grounding | Grounded against References A (SASRec Kang & McAuley 2018), B (SASRecF RecBole), C (DIF-SR), D (TASIF). Recorded distinction: TAIMMA Early Fusion uses element-wise addition (not concatenation). Implemented standard interfaces: `encode_item_sequence`, `encode_context`, `score_candidates`, `score_full_catalog`. | Implemented |
| **2026-10-27** | Fusion Interfaces | Implemented `EarlyFusionSASRec` (supporting 8 source combinations) and `ResidualLateFusionModel` (frozen ID backbone + joint residual encoders + leave-one-source-out utility labels $\Delta_m(q)$). | Verified (21/21 Unit Tests) |
| **2026-10-27** | Complete Pipeline (P1–P5) | Implemented `ResidualLateFusionTrainer` (S1b), RQ1–RQ3 `diagnostics.py`, locked 26/10 `e0_e1_protocol.py` (noise threshold $\epsilon_m$, stability $\rho$, E0/E1, decision engine), `UASFModel` (exact ID recovery when $\alpha=0$, selective contextual gate, 7 baselines), and `statistics.py` (paired bootstrap, Holm correction, TOST). | Verified (37/37 Unit Tests) |


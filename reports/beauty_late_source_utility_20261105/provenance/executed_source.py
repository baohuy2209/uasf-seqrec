"""Minh's post-audit tasks. Requires an already written initial readiness report."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from metadata_pipeline.loader import load_catalog
from taimma.data.ingestion import Amazon2014Ingestor
from taimma.data.split import SequenceSplitter
from taimma.evaluation.evaluator import IndependentEvaluator
from taimma.training.multimodal_trainer import gather_source_sequences
from run_checkpoint1 import Checkpoint

ORDER = ["ID", "C", "B", "T", "CB", "CT", "BT", "CBT"]
SEEDS = [42, 43, 44]
REFS = [42, 43]
RUN = ROOT / "runs/checkpoint1_version1"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def append_report(text):
    with (ROOT / "minh_report.md").open("a", encoding="utf-8") as f:
        f.write("\n\n## Cập nhật " + dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).isoformat() + "\n" + text + "\n")


def ndcg(ranks):
    ranks = np.asarray(ranks)
    assert np.all(ranks >= 1)
    return np.where(ranks <= 10, 1 / np.log2(1 + ranks), 0.0)


def winner(values):
    values = np.asarray(values)
    return int(np.flatnonzero(values.max() - values <= 1e-12)[0])


def groups(samples, seqs, split, pop, length_edges, tail_edges):
    rows = []
    head_count = sorted(pop[1:], reverse=True)[math.ceil(.2 * split.num_items) - 1]
    for s in samples:
        # Preserve actual OOV positions, distinguish them from left padding.
        n = len(seqs[s.user_idx].items)
        end = n - {"meta": 3, "val": 2, "test": 1}[groups.phase]
        raw_history = seqs[s.user_idx].items[max(0, end - 50):end]
        ids = [split.item2idx.get(i, 0) for i in raw_history]
        assert [i for i in ids if i] == [i for i in s.history if i]
        length = sum(i > 0 for i in ids)
        tail = np.mean([i == 0 or pop[i] < head_count for i in ids])
        lg = int(np.searchsorted(length_edges, length, side="right"))
        tg = int(tail >= tail_edges[lg]) if tail_edges is not None else 0
        context_id = hashlib.sha256(json.dumps([s.user_idx, s.history, s.target]).encode()).hexdigest()
        rows.append(dict(split=groups.phase, user_idx=s.user_idx, context_id=context_id,
                         history_len=length, observed_positions=len(ids), history_oov=sum(i == 0 for i in ids),
                         tail_ratio=float(tail), length_group=lg, tail_group=tg, group_id=lg * 2 + tg))
    return pd.DataFrame(rows)


def task1(out, seqs, split, progress):
    p = out / "task1"
    p.mkdir(exist_ok=False)
    # All Train-base interactions, including the first item of each sequence.
    pop = np.zeros(split.num_items + 1, dtype=np.int64)
    for s in seqs:
        for item in s.items[:-3]:
            pop[split.item2idx[item]] += 1
    groups.phase = "meta"
    training = groups(split.train_meta_samples, seqs, split, pop, [], None)
    length_edges = np.quantile(training.history_len, [.2, .4, .6, .8], method="linear")
    training = groups(split.train_meta_samples, seqs, split, pop, length_edges, None)
    tail_edges = [float(training.loc[training.length_group == i, "tail_ratio"].median())
                  if (training.length_group == i).any() else float(training.tail_ratio.median()) for i in range(5)]
    rule = dict(authority="Agent diagnostic proposal predeclared in initial_report.md; not supervisor approval",
                fit_cohort="terminal Train-base histories, before Train-meta target",
                popularity="all Train-base interactions including first item",
                head_boundary_count=int(sorted(pop[1:], reverse=True)[math.ceil(.2 * split.num_items) - 1]),
                head_boundary_ties="include all", length_edges=length_edges.tolist(),
                tail_medians_by_length=tail_edges, edge_ties="higher group", oov="tail",
                empty_validation_group="global validation winner",
                seed_policy="select separately per seed; sample SD ddof=1; repeat excluded",
                subset_ties="within 1e-12; ID,C,B,T,CB,CT,BT,CBT",
                oracle_scope="post-hoc ceiling only for these groups and subsets")
    dump(p / "grouping_rules.json", rule)
    frames = {}
    for phase, samples in [("val", split.val_samples), ("test", split.test_samples)]:
        groups.phase = phase
        frames[phase] = groups(samples, seqs, split, pop, length_edges, tail_edges)
    pd.concat(frames.values()).to_csv(p / "user_context_groups.csv", index=False)
    pd.DataFrame(dict(item_id=np.arange(1, len(pop)), train_base_count=pop[1:],
                      head=pop[1:] >= rule["head_boundary_count"])).to_csv(p / "popularity.csv", index=False)
    metrics, selected, summaries = [], [], []
    for seed in SEEDS:
        scores = {}
        for phase in ["val", "test"]:
            columns = []
            for subset in ORDER:
                key = f"id_seed{seed}" if subset == "ID" else f"late_{subset}_seed{seed}"
                f = pd.read_csv(Path(progress[key]["folder"]) / f"per_user_{phase}.csv").set_index("user_idx")
                assert f.index.is_unique and set(f.index) == set(frames[phase].user_idx)
                columns.append(ndcg(f.loc[frames[phase].user_idx, "rank"]))
            scores[phase] = np.column_stack(columns)
            for group in range(10):
                mask = frames[phase].group_id.to_numpy() == group
                for j, subset in enumerate(ORDER):
                    metrics.append(dict(seed=seed, split=phase, group_id=group, subset=subset,
                                        n_users=int(mask.sum()), ndcg_10=float(scores[phase][mask, j].mean()) if mask.any() else None))
        global_idx = winner(scores["val"].mean(axis=0))
        predictions = np.zeros(len(frames["test"]))
        oracle = np.zeros(len(frames["test"]))
        for group in range(10):
            vm = frames["val"].group_id.to_numpy() == group
            tm = frames["test"].group_id.to_numpy() == group
            chosen = winner(scores["val"][vm].mean(axis=0)) if vm.any() else global_idx
            ceiling = winner(scores["test"][tm].mean(axis=0)) if tm.any() else None
            predictions[tm] = scores["test"][tm, chosen]
            if tm.any():
                oracle[tm] = scores["test"][tm, ceiling]
            selected.append(dict(seed=seed, group_id=group, validation_users=int(vm.sum()),
                                 test_users=int(tm.sum()), chosen_by_validation=ORDER[chosen],
                                 oracle_chosen_by_test=ORDER[ceiling] if ceiling is not None else None,
                                 empty_validation_fallback=not vm.any()))
        for method, value in [("ID", scores["test"][:, 0].mean()),
                              ("global_validation_subset", scores["test"][:, global_idx].mean()),
                              ("group_validation_subset", predictions.mean()),
                              ("group_test_oracle", oracle.mean())]:
            summaries.append(dict(seed=seed, method=method, n_users=len(predictions),
                                  test_ndcg_10=float(value), global_chosen_by_validation=ORDER[global_idx]))
        assert oracle.mean() + 1e-12 >= predictions.mean()
        assert oracle.mean() + 1e-12 >= scores["test"][:, global_idx].mean()
    pd.DataFrame(metrics).to_csv(p / "group_metrics.csv", index=False)
    pd.DataFrame(selected).to_csv(p / "selected_configurations.csv", index=False)
    per_seed = pd.DataFrame(summaries)
    per_seed.to_csv(p / "summary_per_seed.csv", index=False)
    summary = per_seed.groupby("method").test_ndcg_10.agg(["mean", "std"]).reset_index()
    summary.to_csv(p / "summary.csv", index=False)
    append_report("Task1 late: hoàn tất phạm vi chẩn đoán bằng quy tắc agent đã công bố; early vẫn BLOCKED. "
                  f"Output: {p.relative_to(ROOT)}. Mean ± SD mẫu qua seeds42/43/44:\n\n" +
                  "\n".join(f"- {r.method}: {r.mean:.9f} ± {r.std:.9f}" for r in summary.itertuples()) +
                  "\nKhông suy luận ý nghĩa thống kê từ mean/SD. Oracle chỉ hậu kiểm 10 nhóm ×8 cấu hình. "
                  "Nhóm val/test dùng cùng ngưỡng nhưng gán theo lịch sử của từng thời điểm.")
    print(summary.to_string(index=False), flush=True)


def task2(out, split, progress, attrs, masks, device):
    p = out / "task2"
    p.mkdir(exist_ok=False)
    frozen = Checkpoint.__new__(Checkpoint)
    frozen.split, frozen.attrs, frozen.masks = split, attrs, masks
    frozen.progress, frozen.device, frozen.backbone_dim = progress, torch.device(device), 64
    attrs = {k: v.to(device) for k, v in attrs.items()}
    masks = {k: v.to(device) for k, v in masks.items()}
    old = pd.read_csv(RUN / "utility.csv")
    ce_rows, sign_rows, validation_rows, verification = [], [], [], []
    for seed in REFS:
        print(f"Task2 frozen CBT reference {seed}", flush=True)
        model = frozen.load_residual(progress[f"late_CBT_seed{seed}"])
        model.eval()
        model.requires_grad_(False)
        for phase, samples in [("train_meta", [s for s in split.train_meta_samples if s.target]), ("val", split.val_samples)]:
            for start in range(0, len(samples), 256):
                batch = samples[start:start + 256]
                ids = torch.tensor([s.history for s in batch], device=device)
                targets = torch.tensor([s.target for s in batch], device=device)
                users = [s.user_idx for s in batch]
                with torch.no_grad():
                    all_scores, _, branches = model.compute_all_scores(ids, gather_source_sequences(ids, attrs, masks),
                                                                       attrs, masks, apply_dropout=False)
                    ce_all = F.cross_entropy(all_scores, targets - 1, reduction="none").cpu().numpy() if phase == "train_meta" else None
                    all_np = all_scores.cpu().numpy()
                    ev = IndependentEvaluator(split.num_items, mask_policy="input_history_strict")
                    base = ev.evaluate_ranks(all_np, targets.tolist(), users, histories=[s.history for s in batch])
                    for source in ["category", "brand", "title"]:
                        without = all_scores - branches[source] * masks[source].unsqueeze(0)
                        wo = ev.evaluate_ranks(without, targets.tolist(), users, histories=[s.history for s in batch])
                        ce_wo = F.cross_entropy(without, targets - 1, reduction="none").cpu().numpy() if phase == "train_meta" else None
                        for i, s in enumerate(batch):
                            row = dict(reference_seed=seed, user_idx=s.user_idx,
                                       context_id=hashlib.sha256(json.dumps([s.user_idx, s.history, s.target]).encode()).hexdigest(),
                                       target_idx=s.target, source_name=source,
                                       rank_all=base.per_user_records[i].rank, rank_without_m=wo.per_user_records[i].rank)
                            if phase == "train_meta":
                                row.update(ce_all=float(ce_all[i]), ce_without_m=float(ce_wo[i]),
                                           delta_ce=float(ce_wo[i] - ce_all[i]),
                                           delta_logrank=float(np.log2(1 + row["rank_without_m"]) - np.log2(1 + row["rank_all"])))
                                ce_rows.append(row)
                            else:
                                row.update(ndcg_all=base.per_user_records[i].ndcg_10,
                                           ndcg_without_m=wo.per_user_records[i].ndcg_10)
                                validation_rows.append(row)
                if start % 5120 == 0:
                    print(f"  {phase}: {start + len(batch)}/{len(samples)}", flush=True)
        del model
    meta = pd.DataFrame(ce_rows)
    val = pd.DataFrame(validation_rows)
    keys = ["reference_seed", "source_name", "context_id", "user_idx", "target_idx"]
    assert not meta.duplicated(keys).any() and not val.duplicated(keys).any()
    paired = meta.merge(old, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("_new", "_old"))
    assert (paired._merge == "both").all()
    err = float(abs(paired.delta_ce - paired.delta).max())
    assert err < 1e-4, f"CE replay mismatch {err}"
    for seed in REFS:
        stored = pd.read_csv(Path(progress[f"late_CBT_seed{seed}"]["folder"]) / "per_user_val.csv").set_index("user_idx")
        current = val[(val.reference_seed == seed) & (val.source_name == "category")].set_index("user_idx")
        assert np.array_equal(current.loc[stored.index, "rank_all"], stored["rank"]), "all validation ranks changed"
        verification.append(dict(seed=seed, all_validation_ranks_exact=True, ce_replay_max_error=err))
        for source in ["category", "brand", "title"]:
            t = meta[(meta.reference_seed == seed) & (meta.source_name == source)]
            v = val[(val.reference_seed == seed) & (val.source_name == source)]
            sign_rows.append(dict(reference_seed=seed, source_name=source, n_meta=len(t), n_val=len(v),
                                  mean_delta_ce=t.delta_ce.mean(), mean_delta_logrank=t.delta_logrank.mean(),
                                  dval=(v.ndcg_all - v.ndcg_without_m).mean()))
    meta.to_csv(p / "utility_supplement.csv", index=False)
    val.to_csv(p / "validation_same_reference_ablation.csv", index=False)
    by_seed = pd.DataFrame(sign_rows)
    by_seed.to_csv(p / "alignment_per_seed.csv", index=False)
    agg = by_seed.groupby("source_name")[["mean_delta_ce", "mean_delta_logrank", "dval"]].mean().reset_index()
    def sign(v, tolerance):
        return np.where(abs(v) <= tolerance, 0, np.sign(v)).astype(int)
    for col in ["mean_delta_ce", "mean_delta_logrank", "dval"]:
        agg["sign_" + col] = sign(agg[col], 1e-8 if col == "dval" else 1e-6)
    agg["ce_matches"] = (agg.sign_mean_delta_ce == agg.sign_dval) & (agg.sign_dval != 0)
    agg["logrank_matches"] = (agg.sign_mean_delta_logrank == agg.sign_dval) & (agg.sign_dval != 0)
    agg.to_csv(p / "sign_alignment.csv", index=False)
    label = "CE" if agg.ce_matches.all() else "log-rank" if agg.logrank_matches.all() else None
    decision = dict(selected_label=label, status="SELECTED" if label else "BLOCKED",
                    scope="Beauty historical late CBT reference seeds42/43; not architecture approval",
                    seed_aggregation="equal-weight means, predeclared",
                    ce_tolerance=1e-6, logrank_tolerance=1e-6, dval_tolerance=1e-8,
                    rank_mask="input_history_strict", ce_mask="unmasked full catalog as historical artifact",
                    rule="CE if all3 signs match; else log-rank if all3 match; else no label",
                    per_seed_sign_disagreement={source: {col: len(set(sign(g[col], 1e-8 if col == "dval" else 1e-6))) > 1
                                                         for col in ["mean_delta_ce", "mean_delta_logrank", "dval"]}
                                                   for source,g in by_seed.groupby("source_name")},
                    verification=verification)
    dump(p / "label_decision.json", decision)
    append_report(f"Task2: đã suy luận không training, cùng frozen late_CBT42/43. Output {p.relative_to(ROOT)}. "
                  f"CE replay max error={err:.3g}; all validation ranks tái lập chính xác. "
                  f"Nhãn được chọn: {label if label else 'CHƯA ĐẠT — cả CE và log-rank đều không khớp đủ ba nguồn'}. "
                  "Dấu dựa mean equal-weight hai seed, tolerance công bố trước; từng seed và bất đồng lưu riêng. "
                  "Log-rank không tương đương hoàn toàn NDCG@10.\n\n" +
                  "\n".join(f"- {r.source_name}: mean ΔCE={r.mean_delta_ce:.9g}; mean Δlogrank={r.mean_delta_logrank:.9g}; Dval={r.dval:.9g}."
                            for r in agg.itertuples()))
    print(agg.to_string(index=False), flush=True)
    return decision


def task3(out, decision):
    p = out / "task3"
    p.mkdir(exist_ok=False)
    rows = [
        ["log_len", "ln(max(1,known history length)), window50", "observed history", "implemented", "no target; actual log(L), not log1p(L)"],
        ["coverage", "mean source availability over known history items", "catalog availability + observed history", "implemented", "OOV removed as current code; inspect intended definition"],
        ["entropy", "-sum p(label)*ln p(label); multilabel counts; title uses exact dense-vector identity", "observed metadata labels/features", "implemented", "function docstring says unique count but code Shannon entropy; title semantics need approval"],
        ["recency", "mean availability in last3 known history items (all if shorter)", "observed history + source masks", "implemented", "presence measure, not elapsed time; naming requires specification"],
        ["history_popularity", "UNDEFINED in available approved proposal", "Train-base interactions available", "definition missing", "fit popularity on Train-base only; target/future prohibited"],
        ["sequence_state_summary", "UNDEFINED pooling/layer/projection in available proposal", "frozen context embeddings computable from checkpoint", "definition missing", "model choice and summary formula must be approved; no target representation"],
        ["branch_agreement", "UNDEFINED score/rank/similarity/top-k metric in available proposal", "frozen branch scores computable", "definition missing", "target correctness must not enter predictor; OOF transformations train-fold only"],
    ]
    pd.DataFrame(rows, columns=["feature", "formula_or_gap", "data_source", "availability", "leakage_or_semantics"]).to_csv(p / "psi_spec.csv", index=False)
    status = dict(status="PARTIAL", fit_executed=False, completed_scope="psi specification and leakage/gap audit",
                  blockers=["approved reference architecture absent", "full current psi proposal with formulas absent"] +
                           ([] if decision["selected_label"] else ["Task2 no label satisfies all3 sign rule"]),
                  locked_rules=dict(epsilon="median absolute paired-seed delta difference; recompute on chosen scale",
                                    rho=[.1,.3], oof="group by user; all learned preprocessing fit only within training fold",
                                    e1="AUC>=0.55, lower95CI>0.5 and greater than length-only",
                                    metrics="LR, small tree from existing protocol, length baseline, user-cluster bootstrap1000, Spearman with CI"),
                  historical_e1="four features CE only; no final UASF/full-psi inference")
    dump(p / "readiness.json", status)
    append_report(f"Task3 PARTIAL: hoàn tất bảng ψ và leakage tại {p.relative_to(ROOT)}/psi_spec.csv, readiness.json. "
                  "Chưa fit E1/OOF mới. Thiếu: " + "; ".join(status["blockers"]) +
                  ". Không dùng E1 CE bốn feature lịch sử thay cho E1 đầy đủ ψ, không tự định nghĩa các feature còn thiếu. "
                  "Khi đủ đầu vào phải tính lại epsilon/stability trên thang nhãn đã chọn, giữ OOF theo user và fit preprocessing trong train fold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    assert out.is_relative_to(ROOT / "reports") and (out / "initial_report.md").is_file()
    assert "Readiness BAN ĐẦU" in (ROOT / "minh_report.md").read_text()
    torch.set_num_threads(2)
    progress = json.loads((RUN / "progress.json").read_text())
    lock = json.loads((RUN / "protocol.lock.json").read_text())
    assert all(sha(ROOT / k) == v for k, v in lock["code_sha256"].items())
    seqs, _ = Amazon2014Ingestor("data/raw", "Beauty").build_interaction_sequences(include_metadata=False)
    split = SequenceSplitter().split(seqs)
    manifest = json.loads((ROOT / "artifacts/metadata/beauty/manifest.json").read_text())
    assert sha(ROOT / manifest["reviews_file"]) == manifest["reviews_sha256"]
    assert sha(ROOT / manifest["metadata_file"]) == manifest["metadata_sha256"]
    directory = ROOT / "artifacts/metadata/beauty"
    assert all(sha(directory / k) == v for k, v in manifest["artifact_sha256"].items())
    itemmap = pd.read_csv(directory / "item_map.csv")
    catalog_rows = itemmap[itemmap.in_train_base]
    assert dict(zip(catalog_rows.raw_item_id.astype(str), catalog_rows.model_item_id)) == split.item2idx
    assert (itemmap.loc[~itemmap.in_train_base, "model_item_id"] == 0).all()
    config = dict(command=sys.argv, cwd=str(ROOT), code_sha256=sha(__file__),
                  head="7bb443431771aaff98b726fccb043a888eec2163",
                  source_lock_sha256=sha(RUN / "protocol.lock.json"),
                  inputs={str(p.relative_to(ROOT)):sha(p) for p in
                          [RUN/"utility.csv",RUN/"utility.csv.manifest.json",RUN/"utility_cohort.csv",
                           RUN/"results.csv",directory/"manifest.json",out/"initial_report.md"]},
                  seeds=SEEDS, reference_seeds=REFS, device=args.device,
                  catalog_mapping="reconstructed sorted mapping matches metadata item_map; old pilot mapping excluded")
    dump(out / "execution_manifest.json", config)
    task1(out, seqs, split, progress)
    attrs, masks = load_catalog(directory)
    decision = task2(out, split, progress, attrs, masks, args.device)
    task3(out, decision)
    dump(out / "COMPLETED.json", dict(completed_scopes=["task1 late diagnostic","task2 same-reference label alignment","task3 psi gap audit"],
                                      task1="PARTIAL: early grid and approved grouping absent",
                                      task2=decision["status"], task3="PARTIAL; E1 fit blocked"))
    outputs = {str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file() and p.name != "output_checksums.json"}
    dump(out / "output_checksums.json", outputs)


if __name__ == "__main__":
    main()

"""Read-only E1 evidence validation with lossless CSV float parsing.

The legacy exporter/inspector file remains byte-identical because manifests
bind its SHA256. All original hash, label, fold and metric checks are retained.
No predictor fitting, data changes or tolerance relaxation occurs here.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from .memo0711_e1_evidence import sha, _json, _assert_metrics


def inspect_e1_roundtrip(dataset_root: Path) -> dict:
    """Validate paired evidence using hashes/metrics only; never fit predictors."""
    root = Path(dataset_root).resolve()
    checked = []
    for metric in ('CE', 'LOG_RANK'):
        report_path = root / 'selected_s2' / f'e0_e1_{metric}.json'
        marker = root / 'e1_oof_evidence' / metric / 'manifest.json'
        if not report_path.is_file() or not marker.is_file():
            checked.append(dict(metric=metric, status='MISSING_E1_OOF_EVIDENCE'))
            continue
        record = _json(marker)
        if record.get('signature', {}).get('schema_version') != 1:
            raise ValueError('Unknown E1 OOF evidence schema')
        for absolute, expected in record['signature']['sources'].items():
            p = Path(absolute)
            if not p.is_file():
                # A copied bundle may be audited on Windows without changing its
                # original Linux provenance. Rebase only declared dataset/repo roots.
                normalized=str(absolute).replace('\\','/')
                for old, new in [(record['signature'].get('dataset_root'), root),
                                 (record['signature'].get('repository_root'), root.parents[2])]:
                    prefix=str(old).replace('\\','/').rstrip('/')+'/' if old else ''
                    if prefix and normalized.startswith(prefix):
                        candidate=(new/normalized[len(prefix):]).resolve()
                        if not candidate.is_relative_to(new.resolve()):raise ValueError('E1 evidence path escapes declared root')
                        p=candidate;break
            if not p.is_file() or sha(p) != expected:
                raise ValueError('E1 evidence source changed: ' + str(p))
        for relative, expected in record['files'].items():
            p = (marker.parent / relative).resolve()
            if not p.is_relative_to(marker.parent.resolve()) or not p.is_file() or sha(p) != expected:
                raise ValueError('E1 OOF evidence artifact changed')
        summary = _json(marker.parent / 'replay_summary.json')
        reports = _json(report_path)['sources']
        csv_path = marker.parent / 'paired_oof_predictions.csv'
        frame = pd.read_csv(csv_path, float_precision="round_trip") if csv_path.is_file() else pd.DataFrame()
        for item in summary['sources']:
            source = item['source']
            if item['status'] == 'NOT_EVALUABLE':
                if reports[source]['e1'].get('status') == 'EVALUATED':
                    raise ValueError('Evaluable E1 source cannot omit paired predictions')
                continue
            rows = frame.loc[frame.source_name == source]
            if rows.empty or rows.duplicated(['user_idx', 'context_id']).any():
                raise ValueError('Missing or duplicate E1 OOF contexts')
            if rows.groupby('user_idx').fold.nunique().max() != 1 or set(rows.fold) != set(range(5)):
                raise ValueError('OOF users span folds or folds are missing')
            y = rows.label.to_numpy(); probabilities = rows[['probability_lr', 'probability_tree', 'probability_length']].to_numpy()
            if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)) or not np.array_equal(y, (rows.delta > rows.epsilon).astype(int)):
                raise ValueError('Invalid OOF labels or probabilities')
            rho = None if np.ptp(rows.probability_lr) == 0 or np.ptp(rows.delta) == 0 else float(stats.spearmanr(rows.probability_lr, rows.delta).statistic)
            actual = dict(oof_auc_lr=float(roc_auc_score(y, probabilities[:, 0])),
                          oof_auc_tree=float(roc_auc_score(y, probabilities[:, 1])),
                          baseline_len_auc=float(roc_auc_score(y, probabilities[:, 2])), oof_spearman_rho=rho)
            _assert_metrics(actual, reports[source]['e1'])
        checked.append(dict(metric=metric, status='PASS', manifest=str(marker), manifest_sha256=sha(marker)))
    return dict(status='PASS' if all(r['status'] == 'PASS' for r in checked) else 'INCOMPLETE',
                cpu_predictor_fitting=False, checks=checked)

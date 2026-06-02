import { useEffect, useState } from 'react';
import type { ModelInfo } from '../api';
import { fetchModelInfo } from '../api';

function GateRow({ label, value, gate, passed }: { label: string; value: number; gate: number; passed: boolean }) {
  return (
    <div className="metric-row">
      <span className="key">{label}</span>
      <span style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span className="val">{value.toFixed(4)}</span>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>≥ {gate.toFixed(2)}</span>
        <span className={passed ? 'gate-pass' : 'gate-fail'}>{passed ? '✓' : '✗'}</span>
      </span>
    </div>
  );
}

function MetricBlock({ split, metrics }: { split: string; metrics: Record<string, number> }) {
  const rows = [
    ['Stage 1 F1',        'stage1_f1'],
    ['Stage 1 AUC-ROC',   'stage1_auc'],
    ['Stage 1 Precision', 'stage1_precision'],
    ['Stage 1 Recall',    'stage1_recall'],
    ['Stage 1 Threshold', 'stage1_threshold'],
    ['Stage 2 MAE',       'stage2_mae'],
    ['Stage 2 RMSE',      'stage2_rmse'],
  ] as const;

  return (
    <div className="card">
      <div className="card-title">{split} Metrics</div>
      {rows.map(([label, key]) => {
        const v = metrics[key];
        return (
          <div className="metric-row" key={key}>
            <span className="key">{label}</span>
            <span className="val">{v != null && !isNaN(v) ? v.toFixed(4) : '—'}</span>
          </div>
        );
      })}
    </div>
  );
}

export default function ModelStatus() {
  const [info, setInfo]       = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState('');

  useEffect(() => {
    fetchModelInfo()
      .then(setInfo)
      .catch(() => setError('Cannot reach API — is the server running on localhost:8000?'))
      .finally(() => setLoading(false));
  }, []);

  const ts = info
    ? `${info.model_timestamp.slice(0, 4)}-${info.model_timestamp.slice(4, 6)}-${info.model_timestamp.slice(6, 8)} ` +
      `${info.model_timestamp.slice(9, 11)}:${info.model_timestamp.slice(11, 13)}:${info.model_timestamp.slice(13, 15)} UTC`
    : '—';

  return (
    <>
      <div className="page-header">
        <div>
          <h2>Model Status</h2>
          <p>Active model details, evaluation gates, and MLflow experiment link</p>
        </div>
        {info && (
          <div style={{ display: 'flex', gap: 10 }}>
            <span className={`badge ${info.gates.passed ? 'low' : 'high'}`}>
              {info.gates.passed ? '✅ Gates Passed' : '❌ Gates Failed'}
            </span>
          </div>
        )}
      </div>

      <div className="page-body">
        {error && <div className="error-banner">{error}</div>}
        {loading && <div className="spinner" />}

        {info && (
          <>
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-title">Active Model</div>
              <div className="grid-2">
                <div>
                  <div className="metric-row">
                    <span className="key">Trained at</span>
                    <span className="val" style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.82rem' }}>{ts}</span>
                  </div>
                  <div className="metric-row">
                    <span className="key">Stage 1 path</span>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.72rem', color: 'var(--text-code)' }}>
                      {info.stage1_path.split(/[\\\/]/).slice(-1)[0]}
                    </span>
                  </div>
                  <div className="metric-row">
                    <span className="key">Stage 2 (severity)</span>
                    <span className={`badge ${info.stage2_available ? 'low' : 'medium'}`}>
                      {info.stage2_available ? 'Available' : 'Not Available'}
                    </span>
                  </div>
                  <div className="metric-row">
                    <span className="key">Decision threshold</span>
                    <span className="val">{info.threshold.toFixed(4)}</span>
                  </div>
                  <div className="metric-row">
                    <span className="key">Features</span>
                    <span className="val">{info.features.length}</span>
                  </div>
                </div>

                <div>
                  <div className="card-title">Evaluation Gates (Test Set)</div>
                  <GateRow
                    label="F1 Score"
                    value={info.metrics?.test?.stage1_f1 ?? 0}
                    gate={info.gates.f1_gate}
                    passed={(info.metrics?.test?.stage1_f1 ?? 0) >= info.gates.f1_gate}
                  />
                  <GateRow
                    label="AUC-ROC"
                    value={info.metrics?.test?.stage1_auc ?? 0}
                    gate={info.gates.auc_gate}
                    passed={(info.metrics?.test?.stage1_auc ?? 0) >= info.gates.auc_gate}
                  />
                  <div style={{ marginTop: 14, padding: '10px 0', display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontSize: '1.3rem' }}>{info.gates.passed ? '✅' : '❌'}</span>
                    <span style={{ color: info.gates.passed ? 'var(--risk-low)' : 'var(--risk-high)', fontWeight: 600, fontSize: '0.9rem' }}>
                      {info.gates.passed ? 'All gates passed — model approved for production.' : 'Gates failed — model needs improvement.'}
                    </span>
                  </div>

                  <div style={{ marginTop: 12, padding: '10px 14px', background: 'var(--bg-hover)', borderRadius: 'var(--radius-sm)', fontSize: '0.8rem' }}>
                    <span style={{ color: 'var(--text-muted)', marginRight: 8 }}>MLflow UI:</span>
                    <a
                      href="http://localhost:5000"
                      target="_blank"
                      rel="noreferrer"
                      style={{ color: 'var(--accent)', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.78rem' }}
                    >
                      localhost:5000 ↗
                    </a>
                    <div style={{ color: 'var(--text-muted)', marginTop: 4, fontSize: '0.72rem' }}>
                      Run: <code style={{ color: 'var(--text-code)' }}>mlflow ui --backend-store-uri experiments/mlruns</code>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid-2" style={{ marginBottom: 20 }}>
              {info.metrics.val  && <MetricBlock split="Validation" metrics={info.metrics.val}  />}
              {info.metrics.test && <MetricBlock split="Test"       metrics={info.metrics.test} />}
            </div>

            <div className="card">
              <div className="card-title">Feature Set ({info.features.length} features)</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 4 }}>
                {info.features.map(f => (
                  <span key={f} style={{
                    background: 'var(--bg-hover)',
                    border: '1px solid var(--border)',
                    borderRadius: 4,
                    padding: '3px 10px',
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: '0.75rem',
                    color: 'var(--text-code)',
                  }}>{f}</span>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}

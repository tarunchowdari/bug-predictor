import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import type { Prediction } from '../api';
import { fetchFeed, fetchCommitDetail } from '../api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';

function SHAPWaterfall({ shap }: { shap: Record<string, number> }) {
  const entries = Object.entries(shap)
    .map(([feature, value]) => ({ feature, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 15);

  return (
    <div style={{ height: 380 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={entries} layout="vertical" margin={{ left: 140, right: 30 }}>
          <XAxis type="number" domain={['auto', 'auto']} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
          <YAxis
            type="category" dataKey="feature"
            tick={{ fill: 'var(--text-secondary)', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}
            width={130}
          />
          <Tooltip
            formatter={(v: number) => v.toFixed(4)}
            contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8 }}
            labelStyle={{ color: 'var(--text-secondary)' }}
          />
          <ReferenceLine x={0} stroke="var(--border-light)" />
          <Bar dataKey="value" radius={[0, 3, 3, 0]}>
            {entries.map((e, i) => (
              <Cell key={i} fill={e.value >= 0 ? 'rgba(79,142,247,0.75)' : 'rgba(255,77,109,0.75)'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function MetricPill({ label, value, unit = '' }: { label: string; value: string | number; unit?: string }) {
  return (
    <div className="stat-card" style={{ padding: '14px 18px' }}>
      <div className="label">{label}</div>
      <div className="value" style={{ fontSize: '1.4rem' }}>
        {value}<span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginLeft: 4 }}>{unit}</span>
      </div>
    </div>
  );
}

export default function CommitDetail() {
  const { sha } = useParams<{ sha: string }>();
  const nav = useNavigate();
  const [pred, setPred]       = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState('');

  useEffect(() => {
    if (!sha) return;
    setLoading(true);
    fetchCommitDetail(sha)
      .then(setPred)
      .catch(async () => {
        // Fall back to local feed search if the direct endpoint 404s
        try {
          const feed = await fetchFeed(500);
          const found = feed.predictions.find(p => p.commit_sha === sha);
          if (found) setPred(found);
          else setError('Commit not found in current session.');
        } catch {
          setError('Cannot reach API.');
        }
      })
      .finally(() => setLoading(false));
  }, [sha]);

  const riskClass = pred?.risk_level.toLowerCase() ?? '';

  return (
    <>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <button
            onClick={() => nav(-1)}
            style={{ background: 'var(--bg-hover)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', padding: '6px 14px', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '0.82rem' }}
          >
            ← Back
          </button>
          <div>
            <h2>Commit Detail</h2>
            <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.78rem' }}>{sha}</p>
          </div>
        </div>
        {pred && <span className={`badge ${riskClass}`} style={{ fontSize: '0.85rem', padding: '6px 14px' }}>{pred.risk_level} Risk</span>}
      </div>

      <div className="page-body">
        {loading && <div className="spinner" />}
        {error && <div className="error-banner">{error}</div>}

        {pred && (
          <>
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-title">Commit Info</div>
              <p style={{ color: 'var(--text-primary)', fontWeight: 500, marginBottom: 10 }}>
                {pred.commit_message ?? '(no message)'}
              </p>
              <div style={{ display: 'flex', gap: 24, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                <span><b style={{ color: 'var(--text-muted)' }}>Author:</b> {pred.author ?? '—'}</span>
                <span><b style={{ color: 'var(--text-muted)' }}>Repo:</b> {pred.repo ?? '—'}</span>
                <span><b style={{ color: 'var(--text-muted)' }}>Time:</b> {new Date(pred.timestamp).toLocaleString()}</span>
              </div>
            </div>

            <div className="stats-row" style={{ marginBottom: 20 }}>
              <MetricPill label="Bug Probability" value={`${(pred.bug_prob * 100).toFixed(1)}`} unit="%" />
              <MetricPill label="Threshold"       value={`${(pred.threshold * 100).toFixed(1)}`} unit="%" />
              {pred.severity_score != null && (
                <MetricPill label="Severity Score" value={`${(pred.severity_score * 100).toFixed(0)}`} unit="/ 100" />
              )}
              <MetricPill label="Decision" value={pred.is_buggy ? '⚠ Buggy' : '✓ Clean'} />
            </div>

            {pred.feature_shap && Object.keys(pred.feature_shap).length > 0 ? (
              <div className="card">
                <div className="card-title">SHAP Feature Contributions (top 15)</div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 16 }}>
                  Blue bars push toward "buggy", red bars push toward "clean".
                </p>
                <SHAPWaterfall shap={pred.feature_shap} />
              </div>
            ) : (
              <div className="card">
                <div className="card-title">SHAP Feature Contributions</div>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.84rem' }}>
                  SHAP values not available for this prediction.
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}

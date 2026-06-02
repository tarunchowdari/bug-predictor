import type { Prediction } from '../api';
import { repoColor } from '../utils/repoColor';

interface Props { prediction: Prediction; onClick?: () => void; }

function probColor(p: number) {
  if (p >= 0.7) return 'var(--risk-high)';
  if (p >= 0.4) return 'var(--risk-medium)';
  return 'var(--risk-low)';
}

export default function CommitRow({ prediction: p, onClick }: Props) {
  const riskClass = p.risk_level.toLowerCase();
  const ts = new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const dotColor = repoColor(p.repo ?? '');

  return (
    <tr onClick={onClick}>
      <td>
        <span className="commit-sha">{p.commit_sha?.slice(0, 8) ?? '—'}</span>
      </td>
      <td style={{ maxWidth: 220 }}>
        <div className="commit-msg">{p.commit_message ?? '(no message)'}</div>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2, display: 'flex', alignItems: 'center', gap: 5 }}>
          <span>{p.author ?? '—'} ·</span>
          <span
            className="repo-dot"
            style={{ background: dotColor, flexShrink: 0 }}
            title={p.repo ?? ''}
          />
          <span>{p.repo ?? '—'}</span>
        </div>
      </td>
      <td>
        <span className={`badge ${riskClass}`}>{p.risk_level}</span>
      </td>
      <td>
        <div className="prob-bar-wrap">
          <div className="prob-bar-track">
            <div
              className="prob-bar-fill"
              style={{ width: `${p.bug_prob * 100}%`, background: probColor(p.bug_prob) }}
            />
          </div>
          <span className="prob-value">{(p.bug_prob * 100).toFixed(1)}%</span>
        </div>
      </td>
      <td>
        {p.severity_score != null ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className="sev-bar-track">
              <div className="sev-bar-fill" style={{ width: `${p.severity_score * 100}%` }} />
            </div>
            <span className="prob-value">{(p.severity_score * 100).toFixed(0)}</span>
          </div>
        ) : (
          <span style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>—</span>
        )}
      </td>
      <td style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>{ts}</td>
    </tr>
  );
}

import { useEffect, useState, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, BarChart, Bar, Cell,
} from 'recharts';
import type { Prediction } from '../api';
import { fetchRecentPredictions, fetchHealth } from '../api';

type RiskFilter = 'All' | 'High' | 'Medium' | 'Low';

const KAMEI_LABELS: Record<string, string> = {
  NS: 'Subsystems', ND: 'Directories', NF: 'Files', Entropy: 'Entropy',
  LA: 'Lines Added', LD: 'Lines Deleted', LT: 'Lines Total', FIX: 'Is Fix',
  NOD: 'Prior Devs', NUC: 'Prior Commits', AGE: 'Age (days)',
  EXP: 'Author Exp', REXP: 'Recent Exp', SEXP: 'Subsys Exp',
};
const KAMEI_ORDER = ['NS','ND','NF','Entropy','LA','LD','LT','FIX','NOD','NUC','AGE','EXP','REXP','SEXP'];

function probColor(p: number) {
  if (p >= 0.7)  return 'var(--risk-high)';
  if (p >= 0.34) return 'var(--risk-medium)';
  return 'var(--risk-low)';
}

function fmtTs(ts: string) {
  return new Date(ts).toLocaleString([], {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function fmtTime(ts: string) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function ShapBars({ shap }: { shap: Record<string, number> }) {
  const entries = Object.entries(shap)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 10);
  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.001);

  return (
    <div className="shap-list">
      {entries.map(([feat, val]) => (
        <div key={feat} className="shap-row">
          <span className="shap-label">{feat}</span>
          <div className="shap-track">
            <div
              className="shap-fill"
              style={{
                width: `${(Math.abs(val) / maxAbs) * 100}%`,
                background: val > 0 ? 'rgba(255,77,109,0.8)' : 'rgba(45,212,160,0.8)',
              }}
            />
          </div>
          <span className="shap-val" style={{ color: val > 0 ? 'var(--risk-high)' : 'var(--risk-low)' }}>
            {val > 0 ? '+' : ''}{val.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}

function KameiGrid({ metrics }: { metrics: Record<string, number> }) {
  const pairs = KAMEI_ORDER
    .filter(k => k in metrics)
    .map(k => [k, metrics[k]] as [string, number]);

  return (
    <div className="kamei-grid">
      {pairs.map(([key, val]) => (
        <div key={key} className="kamei-row">
          <span className="kamei-key">{KAMEI_LABELS[key] ?? key}</span>
          <span className="kamei-val">{val % 1 === 0 ? val.toFixed(0) : val.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

function ExpandedRow({ p }: { p: Prediction }) {
  return (
    <tr>
      <td colSpan={6} className="expanded-td">
        <div className="expanded-content">
          <div className="expanded-meta">
            <div className="expanded-full-msg">{p.commit_message ?? '(no message)'}</div>
            <div className="expanded-submeta">
              {p.author && <span>Author: <strong>{p.author}</strong> · </span>}
              <span>{fmtTs(p.timestamp)}</span>
              {p.warning && (
                <span style={{ marginLeft: 10 }}>
                  <span className="warning-pill">⚠ {p.warning}</span>
                </span>
              )}
            </div>
          </div>

          {p.kamei_metrics && Object.keys(p.kamei_metrics).length > 0 ? (
            <div>
              <div className="section-title" style={{ marginBottom: 10 }}>Kamei Metrics</div>
              <KameiGrid metrics={p.kamei_metrics} />
            </div>
          ) : (
            <div>
              <div className="section-title" style={{ marginBottom: 8 }}>Kamei Metrics</div>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Not available for this commit</span>
            </div>
          )}

          {p.feature_shap ? (
            <div>
              <div className="section-title" style={{ marginBottom: 10 }}>
                SHAP — Top 10 Features
                <span style={{ marginLeft: 8, fontWeight: 400, color: 'var(--text-muted)', fontSize: '0.7rem' }}>
                  red = toward buggy · green = toward clean
                </span>
              </div>
              <ShapBars shap={p.feature_shap} />
            </div>
          ) : (
            <div>
              <div className="section-title" style={{ marginBottom: 8 }}>SHAP Explanation</div>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Not available</span>
            </div>
          )}
        </div>
      </td>
    </tr>
  );
}

export default function ProjectDetail() {
  const { repoName } = useParams<{ repoName: string }>();
  const repo = decodeURIComponent(repoName ?? '');
  const nav  = useNavigate();

  const [all, setAll]         = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState('');
  const [online, setOnline]   = useState<boolean | null>(null);
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('All');
  const [expandedSha, setExpandedSha] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [preds, health] = await Promise.all([
        fetchRecentPredictions(500),
        fetchHealth(),
      ]);
      setAll(
        preds
          .filter(p => p.repo === repo)
          .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      );
      setOnline(health);
      setError('');
    } catch {
      setError('Cannot reach API');
      setOnline(false);
    } finally {
      setLoading(false);
    }
  }, [repo]);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 10_000);
    return () => clearInterval(id);
  }, [refresh]);

  const filtered = all.filter(p => riskFilter === 'All' || p.risk_level === riskFilter);
  const high     = all.filter(p => p.risk_level === 'High').length;
  const medium   = all.filter(p => p.risk_level === 'Medium').length;
  const low      = all.filter(p => p.risk_level === 'Low').length;

  const toggleExpand = (sha: string | null) =>
    setExpandedSha(prev => prev === sha ? null : sha);

  const showAnalytics = all.length >= 3;

  const lineData = [...all].reverse().map(p => ({
    time: fmtTime(p.timestamp),
    prob: +(p.bug_prob * 100).toFixed(1),
  }));

  const barData = [
    { risk: 'High',   count: high,   fill: '#ff4d6d' },
    { risk: 'Medium', count: medium, fill: '#ff9f40' },
    { risk: 'Low',    count: low,    fill: '#2dd4a0' },
  ].filter(d => d.count > 0);

  return (
    <div className="page-wrap">
      <nav className="top-nav">
        <span className="nav-brand">bug-predictor</span>
        <div className="nav-status">
          <span className={`status-dot ${online === null ? '' : online ? 'online' : 'offline'}`} />
          {online === null ? 'Connecting…' : online ? 'Model Active' : 'Model Offline'}
        </div>
      </nav>

      <div className="page-wrap">
        <div className="page-content">
          <button className="back-btn" onClick={() => nav('/')}>
            ← Back to Projects
          </button>

          <div className="detail-header">
            <div className="detail-repo">{repo}</div>
          </div>

          <div className="stats-row">
            {(['High', 'Medium', 'Low'] as const).map(level => {
              const count = level === 'High' ? high : level === 'Medium' ? medium : low;
              return (
                <div
                  key={level}
                  className="stat-card"
                  style={{ cursor: 'pointer' }}
                  onClick={() => setRiskFilter(f => f === level ? 'All' : level)}
                >
                  <div className="label">{level} Risk</div>
                  <div className="value" style={{ color: level === 'High' ? 'var(--risk-high)' : level === 'Medium' ? 'var(--risk-medium)' : 'var(--risk-low)' }}>
                    {count}
                  </div>
                  <div className="sub">
                    {all.length > 0 ? `${((count / all.length) * 100).toFixed(0)}% of total` : '—'}
                  </div>
                </div>
              );
            })}
            <div className="stat-card">
              <div className="label">Total Predictions</div>
              <div className="value">{all.length}</div>
              <div className="sub">for this project</div>
            </div>
          </div>

          {error && <div className="error-banner">{error}</div>}

          <div className="section-row">
            <span className="section-title">Commit Risk Feed</span>
            <div className="tabs">
              {(['All', 'High', 'Medium', 'Low'] as RiskFilter[]).map(f => (
                <button
                  key={f}
                  className={`tab-btn ${riskFilter === f ? 'active' : ''}`}
                  onClick={() => setRiskFilter(f)}
                >{f}</button>
              ))}
            </div>
          </div>

          <div className="card" style={{ padding: 0 }}>
            {loading ? (
              <div className="spinner" />
            ) : filtered.length === 0 ? (
              <div className="empty-state" style={{ padding: '48px 32px' }}>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
                  {all.length === 0 ? 'No commits yet for this project' : 'No commits match the selected filter'}
                </p>
              </div>
            ) : (
              <div className="feed-table-wrap">
                <table className="feed-table">
                  <thead>
                    <tr>
                      <th>SHA</th><th>Commit</th><th>Risk</th><th>Bug Prob</th><th>Severity</th><th>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((p, i) => {
                      const sha        = p.commit_sha ?? `row-${i}`;
                      const isExpanded = expandedSha === sha;
                      return [
                        <tr
                          key={sha}
                          className={`data-row ${isExpanded ? 'expanded-parent' : ''}`}
                          onClick={() => toggleExpand(sha)}
                        >
                          <td><span className="commit-sha">{sha.slice(0, 8)}</span></td>
                          <td style={{ maxWidth: 280 }}>
                            <div className="commit-msg">{p.commit_message ?? '(no message)'}</div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>
                              {p.author ?? '—'}
                            </div>
                          </td>
                          <td><span className={`badge ${p.risk_level.toLowerCase()}`}>{p.risk_level}</span></td>
                          <td>
                            <div className="prob-bar-wrap">
                              <div className="prob-bar-track">
                                <div className="prob-bar-fill" style={{ width: `${p.bug_prob * 100}%`, background: probColor(p.bug_prob) }} />
                              </div>
                              <span className="prob-value">{(p.bug_prob * 100).toFixed(1)}%</span>
                            </div>
                          </td>
                          <td>
                            {p.severity_score != null ? (
                              <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                                <div className="sev-bar-track">
                                  <div className="sev-bar-fill" style={{ width: `${p.severity_score * 100}%` }} />
                                </div>
                                <span className="prob-value">{(p.severity_score * 100).toFixed(0)}</span>
                              </div>
                            ) : (
                              <span style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>—</span>
                            )}
                          </td>
                          <td style={{ color: 'var(--text-muted)', fontSize: '0.78rem', whiteSpace: 'nowrap' }}>
                            {fmtTime(p.timestamp)}
                          </td>
                        </tr>,
                        isExpanded && <ExpandedRow key={`${sha}-exp`} p={p} />,
                      ];
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {showAnalytics && (
            <div className="analytics-section">
              <div className="section-row" style={{ marginBottom: 16 }}>
                <span className="section-title">Project Analytics</span>
              </div>

              <div className="analytics-grid">
                <div className="card">
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: 16, fontWeight: 600 }}>
                    Bug Probability Over Time
                  </div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={lineData} margin={{ top: 4, right: 16, bottom: 4, left: -20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis
                        dataKey="time"
                        tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                        axisLine={false} tickLine={false}
                      />
                      <YAxis
                        domain={[0, 100]}
                        tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                        axisLine={false} tickLine={false}
                        tickFormatter={v => `${v}%`}
                      />
                      <Tooltip
                        contentStyle={{
                          background: 'var(--bg-card)',
                          border: '1px solid var(--border-light)',
                          borderRadius: 6,
                          fontSize: 12,
                          fontFamily: 'JetBrains Mono',
                          color: 'var(--text-primary)',
                        }}
                        formatter={(v: number) => [`${v}%`, 'Bug Prob']}
                      />
                      <Line
                        type="monotone" dataKey="prob"
                        stroke="var(--accent)" strokeWidth={2}
                        dot={{ r: 3, fill: 'var(--accent)', strokeWidth: 0 }}
                        activeDot={{ r: 5 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="card">
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: 16, fontWeight: 600 }}>
                    Risk Distribution
                  </div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={barData} margin={{ top: 4, right: 8, bottom: 4, left: -20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="risk" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
                      <Tooltip
                        contentStyle={{
                          background: 'var(--bg-card)',
                          border: '1px solid var(--border-light)',
                          borderRadius: 6,
                          fontSize: 12,
                          color: 'var(--text-primary)',
                        }}
                        cursor={{ fill: 'rgba(255,255,255,0.04)' }}
                      />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                        {barData.map((d, i) => <Cell key={i} fill={d.fill} fillOpacity={0.85} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

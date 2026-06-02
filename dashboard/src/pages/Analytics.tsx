import { useEffect, useState } from 'react';
import type { Analytics, ShapFeature } from '../api';
import { fetchAnalytics, fetchShapFeatures } from '../api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, CartesianGrid, PieChart, Pie,
} from 'recharts';

const RISK_COLORS: Record<string, string> = {
  High:   'var(--risk-high)',
  Medium: 'var(--risk-medium)',
  Low:    'var(--risk-low)',
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-title">{title}</div>
      {children}
    </div>
  );
}

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [shap, setShap]           = useState<ShapFeature[]>([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState('');

  useEffect(() => {
    Promise.all([fetchAnalytics(), fetchShapFeatures()])
      .then(([a, s]) => {
        setAnalytics(a);
        setShap(s.features.slice(0, 12));
      })
      .catch(() => setError('Cannot reach API — is the server running on localhost:8000?'))
      .finally(() => setLoading(false));
  }, []);

  const riskData = analytics
    ? Object.entries(analytics.by_risk).map(([name, value]) => ({ name, value }))
    : [];

  return (
    <>
      <div className="page-header">
        <div>
          <h2>Analytics</h2>
          <p>Bug rate trends, author risk profiles, and model feature importance</p>
        </div>
      </div>

      <div className="page-body">
        {error && <div className="error-banner">{error}</div>}
        {loading && <div className="spinner" />}

        {analytics && (
          <>
            <div className="stats-row" style={{ marginBottom: 24 }}>
              <div className="stat-card">
                <div className="label">Total Commits</div>
                <div className="value">{analytics.total}</div>
              </div>
              <div className="stat-card">
                <div className="label">Predicted Buggy</div>
                <div className="value" style={{ color: 'var(--risk-high)' }}>{analytics.buggy}</div>
              </div>
              <div className="stat-card">
                <div className="label">Bug Rate</div>
                <div className="value" style={{ color: 'var(--risk-medium)' }}>
                  {(analytics.bug_rate * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            <div className="grid-2">
              <Section title="Bug Rate Over Time">
                {analytics.trends.length > 0 ? (
                  <div style={{ height: 220 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={analytics.trends}>
                        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                        <XAxis dataKey="hour" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                        <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                        <Tooltip
                          formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, 'Bug Rate']}
                          contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8 }}
                        />
                        <Line type="monotone" dataKey="bug_rate" stroke="var(--accent)" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.84rem' }}>No trend data yet.</p>
                )}
              </Section>

              <Section title="Risk Distribution">
                <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {riskData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={riskData} dataKey="value" nameKey="name"
                          cx="50%" cy="50%" outerRadius={80}
                          label={({ name, value }) => `${name}: ${value}`} labelLine={false}
                        >
                          {riskData.map((entry, i) => (
                            <Cell key={i} fill={RISK_COLORS[entry.name] ?? 'var(--accent)'} />
                          ))}
                        </Pie>
                        <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8 }} />
                      </PieChart>
                    </ResponsiveContainer>
                  ) : (
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.84rem' }}>No predictions yet.</p>
                  )}
                </div>
              </Section>
            </div>

            {analytics.top_authors.length > 0 && (
              <Section title="Top Authors by Bug Rate">
                <div style={{ height: 240 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={analytics.top_authors.slice(0, 8)} margin={{ left: 20 }}>
                      <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                      <XAxis dataKey="author" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                      <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                      <Tooltip
                        formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, 'Bug Rate']}
                        contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8 }}
                      />
                      <Bar dataKey="bug_rate" radius={[4, 4, 0, 0]}>
                        {analytics.top_authors.map((entry, i) => (
                          <Cell key={i} fill={
                            entry.bug_rate > 0.5 ? 'var(--risk-high)'
                              : entry.bug_rate > 0.3 ? 'var(--risk-medium)'
                              : 'var(--risk-low)'
                          } />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Section>
            )}
          </>
        )}

        {shap.length > 0 && (
          <div className="card">
            <div className="card-title">Feature Importance — Mean |SHAP| (test set)</div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 16 }}>
              Average absolute SHAP value over 25,959 test commits. Higher = more influence on predictions.
            </p>
            <div style={{ height: 340 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={shap} layout="vertical" margin={{ left: 160, right: 30 }}>
                  <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                  <YAxis
                    type="category" dataKey="feature"
                    tick={{ fill: 'var(--text-secondary)', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}
                    width={155}
                  />
                  <Tooltip
                    formatter={(v: number) => v.toFixed(4)}
                    contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8 }}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {shap.map((_, i) => (
                      <Cell key={i} fill={`hsl(${210 + i * 8}, 80%, ${60 - i * 2}%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

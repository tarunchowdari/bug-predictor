import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import type { Prediction } from '../api';
import { fetchRecentPredictions, fetchHealth } from '../api';

function timeAgo(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1)  return 'just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function probColor(p: number): string {
  if (p >= 0.7)  return 'var(--risk-high)';
  if (p >= 0.34) return 'var(--risk-medium)';
  return 'var(--risk-low)';
}

interface ProjectSummary {
  repo:     string;
  total:    number;
  high:     number;
  medium:   number;
  low:      number;
  avgProb:  number;
  latestTs: string;
}

function groupByRepo(predictions: Prediction[]): ProjectSummary[] {
  const map = new Map<string, Prediction[]>();
  for (const p of predictions) {
    const r = p.repo ?? '(unknown)';
    if (!map.has(r)) map.set(r, []);
    map.get(r)!.push(p);
  }
  return Array.from(map.entries())
    .map(([repo, preds]) => ({
      repo,
      total:    preds.length,
      high:     preds.filter(p => p.risk_level === 'High').length,
      medium:   preds.filter(p => p.risk_level === 'Medium').length,
      low:      preds.filter(p => p.risk_level === 'Low').length,
      avgProb:  preds.reduce((s, p) => s + p.bug_prob, 0) / preds.length,
      latestTs: preds.reduce(
        (latest, p) => new Date(p.timestamp) > new Date(latest) ? p.timestamp : latest,
        preds[0].timestamp
      ),
    }))
    .sort((a, b) => new Date(b.latestTs).getTime() - new Date(a.latestTs).getTime());
}

export default function ProjectSelection() {
  const nav = useNavigate();
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState('');
  const [online, setOnline]     = useState<boolean | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [preds, health] = await Promise.all([
        fetchRecentPredictions(500),
        fetchHealth(),
      ]);
      setProjects(groupByRepo(preds));
      setOnline(health);
      setError('');
    } catch {
      setError('Cannot reach API — is the server running on localhost:8000?');
      setOnline(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, [refresh]);

  const goToProject = (repo: string) => nav('/project/' + encodeURIComponent(repo));

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
          <div className="page-header">
            <h1>Projects</h1>
            <p>Select a project to view commit risk analysis</p>
          </div>

          {error && <div className="error-banner">{error}</div>}

          {loading ? (
            <div className="spinner" />
          ) : projects.length === 0 ? (
            <div className="empty-state">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.2}>
                <path d="M3 7h18M3 12h18M3 17h12" strokeLinecap="round" />
                <circle cx="19" cy="17" r="3" />
                <path d="M19 15.5v1.5l1 1" strokeLinecap="round" />
              </svg>
              <h3>No projects yet</h3>
              <p>
                Connect a GitHub repository and push a commit to get started.
                Add this webhook URL in GitHub Settings → Webhooks:
              </p>
              <div className="webhook-code">
                <div><span className="dim">Method: </span>POST</div>
                <div><span className="dim">URL:    </span>http://&lt;your-server&gt;:8000/webhook/github</div>
                <div><span className="dim">Content-Type: </span>application/json</div>
                <div><span className="dim">Events: </span>Just the push event</div>
              </div>
            </div>
          ) : (
            <div className="project-grid">
              {projects.map(proj => {
                const total     = proj.total || 1;
                const highPct   = (proj.high   / total) * 100;
                const mediumPct = (proj.medium / total) * 100;
                const lowPct    = (proj.low    / total) * 100;
                const avgColor  = probColor(proj.avgProb);

                return (
                  <div
                    key={proj.repo}
                    className="project-card"
                    onClick={() => goToProject(proj.repo)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => e.key === 'Enter' && goToProject(proj.repo)}
                  >
                    <div className="project-card-repo" title={proj.repo}>
                      {proj.repo}
                    </div>

                    <div className="project-card-meta">
                      <div>
                        <div className="project-card-count">{proj.total}</div>
                        <div className="project-card-count-label">predictions</div>
                      </div>
                      <div className="project-card-avg">
                        <div className="project-card-avg-val" style={{ color: avgColor }}>
                          {(proj.avgProb * 100).toFixed(1)}%
                        </div>
                        <div className="project-card-avg-label">avg risk</div>
                      </div>
                    </div>

                    <div className="risk-bar">
                      {proj.high   > 0 && <div className="risk-bar-seg" style={{ width: `${highPct}%`,   background: 'var(--risk-high)'   }} />}
                      {proj.medium > 0 && <div className="risk-bar-seg" style={{ width: `${mediumPct}%`, background: 'var(--risk-medium)' }} />}
                      {proj.low    > 0 && <div className="risk-bar-seg" style={{ width: `${lowPct}%`,    background: 'var(--risk-low)'    }} />}
                    </div>

                    <div className="risk-bar-legend">
                      {proj.high   > 0 && <span className="hl">High {proj.high}</span>}
                      {proj.medium > 0 && <span className="ml">Med {proj.medium}</span>}
                      {proj.low    > 0 && <span className="ll">Low {proj.low}</span>}
                    </div>

                    <div className="project-card-footer">
                      <span>Last commit: {timeAgo(proj.latestTs)}</span>
                      <span style={{ color: 'var(--accent)', fontSize: '0.72rem' }}>View details →</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

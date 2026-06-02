import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import type { Prediction } from '../api';
import { fetchFeed } from '../api';
import CommitRow from '../components/CommitRow';
import { repoColor } from '../utils/repoColor';

type RiskFilter = 'All' | 'High' | 'Medium' | 'Low';

interface Props {
  repoFilter: string;
  setRepoFilter: (r: string) => void;
}

export default function LiveFeed({ repoFilter, setRepoFilter }: Props) {
  const nav = useNavigate();
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState('');
  const [riskFilter, setRiskFilter]   = useState<RiskFilter>('All');
  const [groupByRepo, setGroupByRepo] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await fetchFeed(200);
      setPredictions(data.predictions);
      setError('');
    } catch {
      setError('Cannot reach API — is the server running on localhost:8000?');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 10_000);
    return () => clearInterval(id);
  }, [load]);

  const allRepos = Array.from(
    new Set(predictions.map(p => p.repo ?? '').filter(Boolean))
  ).sort();

  const filtered = predictions
    .filter(p => repoFilter === 'All' || p.repo === repoFilter)
    .filter(p => riskFilter === 'All' || p.risk_level === riskFilter)
    .sort((a, b) => b.bug_prob - a.bug_prob);

  const repoFiltered = repoFilter === 'All'
    ? predictions
    : predictions.filter(p => p.repo === repoFilter);

  const isFiltered = repoFilter !== 'All';

  type RepoGroup = { repo: string; items: Prediction[]; avgProb: number };
  const groups: RepoGroup[] = groupByRepo
    ? Array.from(new Set(filtered.map(p => p.repo ?? '(unknown)'))).map(repo => {
        const items = filtered.filter(p => (p.repo ?? '(unknown)') === repo);
        const avgProb = items.reduce((s, p) => s + p.bug_prob, 0) / items.length;
        return { repo, items, avgProb };
      }).sort((a, b) => b.avgProb - a.avgProb)
    : [];

  return (
    <>
      <div className="page-header" style={{ flexWrap: 'wrap', gap: 10 }}>
        <div>
          <h2><span className="live-dot" />Live Feed</h2>
          <p>Commit risk predictions, sorted by probability — auto-refreshes every 10 s</p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <div style={{ position: 'relative' }}>
            <select
              id="repo-filter"
              className="repo-select"
              value={repoFilter}
              onChange={e => setRepoFilter(e.target.value)}
            >
              <option value="All">All Repos</option>
              {allRepos.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
            {isFiltered && (
              <span className="repo-select-dot" style={{ background: repoColor(repoFilter) }} />
            )}
          </div>

          <button
            id="group-by-toggle"
            className={`tab-btn ${groupByRepo ? 'active' : ''}`}
            style={{ fontSize: '0.78rem', padding: '4px 12px' }}
            onClick={() => setGroupByRepo(g => !g)}
            title="Group predictions by repository"
          >
            ⊞ Group by Repo
          </button>

          <div className="tabs">
            {(['All', 'High', 'Medium', 'Low'] as RiskFilter[]).map(f => (
              <button
                key={f}
                id={`risk-filter-${f.toLowerCase()}`}
                className={`tab-btn ${riskFilter === f ? 'active' : ''}`}
                onClick={() => setRiskFilter(f)}
              >{f}</button>
            ))}
          </div>
        </div>
      </div>

      <div className="page-body">
        {error && <div className="error-banner">{error}</div>}

        <div className="stats-row">
          {(['High', 'Medium', 'Low'] as const).map(level => {
            const count = repoFiltered.filter(p => p.risk_level === level).length;
            const total = repoFiltered.length;
            return (
              <div
                key={level}
                className={`stat-card ${isFiltered ? 'filtered' : ''}`}
                onClick={() => setRiskFilter(level)}
                style={{ cursor: 'pointer' }}
              >
                <div className="label">
                  {level} Risk
                  {isFiltered && <span className="filtered-pill">filtered</span>}
                </div>
                <div className="value" style={{ color: level === 'High' ? 'var(--risk-high)' : level === 'Medium' ? 'var(--risk-medium)' : 'var(--risk-low)' }}>
                  {count}
                </div>
                <div className="sub">
                  {total > 0 ? `${((count / total) * 100).toFixed(0)}% of ${isFiltered ? repoFilter.split('/').pop() : 'total'}` : '—'}
                </div>
              </div>
            );
          })}
          <div className={`stat-card ${isFiltered ? 'filtered' : ''}`}>
            <div className="label">
              Total Predictions
              {isFiltered && <span className="filtered-pill">filtered</span>}
            </div>
            <div className="value">{repoFiltered.length}</div>
            <div className="sub">{isFiltered ? `of ${predictions.length} total` : 'in session'}</div>
          </div>
        </div>

        <div className="card">
          {loading ? (
            <div className="spinner" />
          ) : filtered.length === 0 ? (
            <div className="empty-state">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}><path d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
              <p>No predictions match the current filters.</p>
            </div>
          ) : groupByRepo ? (
            <div>
              {groups.map(g => (
                <div key={g.repo} className="repo-group">
                  <div className="repo-group-header">
                    <span className="repo-dot" style={{ background: repoColor(g.repo) }} />
                    <span className="repo-group-name">{g.repo}</span>
                    <span className="repo-group-meta">
                      {g.items.length} commit{g.items.length !== 1 ? 's' : ''}
                    </span>
                    <span className="repo-group-avg" style={{ color: g.avgProb >= 0.7 ? 'var(--risk-high)' : g.avgProb >= 0.34 ? 'var(--risk-medium)' : 'var(--risk-low)' }}>
                      avg {(g.avgProb * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="feed-table-wrap">
                    <table className="feed-table">
                      <thead>
                        <tr><th>SHA</th><th>Commit</th><th>Risk</th><th>Bug Prob</th><th>Severity</th><th>Time</th></tr>
                      </thead>
                      <tbody>
                        {g.items.map((p, i) => (
                          <CommitRow
                            key={`${p.commit_sha}-${i}`}
                            prediction={p}
                            onClick={() => p.commit_sha && nav(`/commit/${p.commit_sha}`)}
                          />
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="feed-table-wrap">
              <table className="feed-table">
                <thead>
                  <tr><th>SHA</th><th>Commit</th><th>Risk</th><th>Bug Prob</th><th>Severity</th><th>Time</th></tr>
                </thead>
                <tbody>
                  {filtered.map((p, i) => (
                    <CommitRow
                      key={`${p.commit_sha}-${i}`}
                      prediction={p}
                      onClick={() => p.commit_sha && nav(`/commit/${p.commit_sha}`)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

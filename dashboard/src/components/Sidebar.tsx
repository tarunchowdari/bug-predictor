import { useNavigate, useLocation } from 'react-router-dom';

const LINKS = [
  {
    to: '/',
    label: 'Live Feed',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
      </svg>
    ),
  },
  {
    to: '/analytics',
    label: 'Analytics',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
        <line x1="8" y1="21" x2="16" y2="21" />
        <line x1="12" y1="17" x2="12" y2="21" />
      </svg>
    ),
  },
  {
    to: '/model',
    label: 'Model Status',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14" />
      </svg>
    ),
  },
];

export default function Sidebar() {
  const nav = useNavigate();
  const loc = useLocation();

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h1>bug-predictor</h1>
        <span>v1.0 · XGBoost</span>
      </div>

      <nav className="sidebar-nav">
        {LINKS.map(l => (
          <div
            key={l.to}
            className={`nav-item ${loc.pathname === l.to ? 'active' : ''}`}
            onClick={() => nav(l.to)}
          >
            {l.icon}
            <span>{l.label}</span>
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span className="live-dot" />
        API · localhost:8000
      </div>
    </aside>
  );
}

import axios from 'axios';

export const BASE_URL = 'http://localhost:8000';
const api = axios.create({ baseURL: BASE_URL, timeout: 8000 });

export interface Prediction {
  commit_sha:     string | null;
  repo:           string | null;
  author:         string | null;
  commit_message: string | null;
  bug_prob:       number;
  is_buggy:       boolean;
  risk_level:     'High' | 'Medium' | 'Low';
  severity_score: number | null;
  threshold:      number;
  timestamp:      string;
  feature_shap:   Record<string, number> | null;
  kamei_metrics?: Record<string, number>;
  warning?:       string;
}

export interface ModelInfo {
  model_timestamp:  string;
  stage1_path:      string;
  stage2_available: boolean;
  threshold:        number;
  features:         string[];
  metrics: {
    val?:  Record<string, number>;
    test?: Record<string, number>;
  };
  gates: {
    f1_gate:  number;
    auc_gate: number;
    passed:   boolean;
  };
}

export interface Analytics {
  total:       number;
  buggy:       number;
  bug_rate:    number;
  by_risk:     Record<string, number>;
  top_authors: { author: string; total: number; bug_rate: number }[];
  trends:      { hour: string; total: number; buggy: number; bug_rate: number }[];
}

export interface ShapFeature {
  feature:    string;
  importance: number;
}

// Primary endpoints used by the current dashboard
export const fetchRecentPredictions = (limit = 500): Promise<Prediction[]> =>
  api.get<{ count: number; predictions: Prediction[] }>('/predictions/recent', { params: { limit } })
     .then(r => r.data.predictions);

export const fetchHealth = (): Promise<boolean> =>
  api.get('/health', { timeout: 3000 }).then(() => true).catch(() => false);

// Legacy endpoints
export const fetchFeed = (limit = 100) =>
  api.get<{ count: number; predictions: Prediction[] }>('/feed', { params: { limit } }).then(r => r.data);

export const fetchCommitDetail = (sha: string) =>
  api.get<Prediction>(`/feed/${sha}`).then(r => r.data);

export const fetchModelInfo = () =>
  api.get<ModelInfo>('/model/info').then(r => r.data);

export const fetchAnalytics = () =>
  api.get<Analytics>('/analytics').then(r => r.data);

export const fetchShapFeatures = () =>
  api.get<{ features: ShapFeature[] }>('/shap/features').then(r => r.data);

export const submitPredict = (payload: Record<string, unknown>) =>
  api.post<Prediction>('/predict', payload).then(r => r.data);

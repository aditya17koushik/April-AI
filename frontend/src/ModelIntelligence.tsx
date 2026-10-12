import { useEffect, useMemo, useState } from 'react';
import { Activity, ArrowRight, BrainCircuit, CheckCircle2, ChevronDown, CircleDot, Cpu, LoaderCircle, Play, RefreshCw, Sparkles, Target, TrendingUp, Zap } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
type Importance = { feature: string; importance: number };
type ModelItem = {
  model_id: string; model: string; task: string; target: string; feature_columns: string[];
  feature_types: Record<string, string>; feature_count: number; rows_used: number; created_at: string;
  feature_importance: Importance[]; results: { model: string; status: string; metrics: Record<string, unknown> }[];
};
type Prediction = { prediction: unknown; task: string; target: string; probabilities?: { class: string; probability: number }[] };
const prettyDate = (value: string) => value ? new Date(value).toLocaleString() : 'Saved model';

export default function ModelIntelligence() {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [selectedId, setSelectedId] = useState('');
  const [values, setValues] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const selected = models.find((item) => item.model_id === selectedId) ?? models[0];

  async function refresh() {
    setLoading(true); setError('');
    try {
      const response = await fetch(`${API_BASE}/api/models`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? 'Could not load saved models.');
      const next = (payload.models ?? []) as ModelItem[];
      setModels(next);
      setSelectedId((current) => next.some((item) => item.model_id === current) ? current : next[0]?.model_id ?? '');
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not load the model registry.'); }
    finally { setLoading(false); }
  }
  useEffect(() => { void refresh(); }, []);
  useEffect(() => { setValues({}); setPrediction(null); }, [selectedId]);

  const metrics = useMemo(() => {
    if (!selected) return [];
    const best = selected.results.find((item) => item.model === selected.model && item.status === 'success');
    if (!best) return [];
    const entries = Object.entries(best.metrics).filter(([key, value]) => key !== 'confusion_matrix' && typeof value === 'number') as [string, number][];
    return entries.slice(0, 4);
  }, [selected]);

  async function runPrediction() {
    if (!selected) return;
    const missing = selected.feature_columns.filter((name) => values[name] === undefined || values[name].trim() === '');
    if (missing.length) { setError(`Enter a value for: ${missing.slice(0, 3).join(', ')}${missing.length > 3 ? '…' : ''}`); return; }
    setBusy(true); setError(''); setPrediction(null);
    const payload: Record<string, unknown> = {};
    selected.feature_columns.forEach((name) => {
      const value = values[name];
      payload[name] = selected.feature_types[name] === 'number' ? Number(value) : value;
    });
    try {
      const response = await fetch(`${API_BASE}/api/models/${selected.model_id}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? 'Prediction failed.');
      setPrediction(data as Prediction);
    } catch (e) { setError(e instanceof Error ? e.message : 'Prediction failed.'); }
    finally { setBusy(false); }
  }

  return <section className="intelligence-studio" id="intelligence">
    <div className="intelligence-hero"><div className="intelligence-copy"><div className="intelligence-eyebrow"><Sparkles size={14}/> PHASE 05 · MODEL INTELLIGENCE</div><h2>From trained to <span>trusted.</span></h2><p>Inspect what drives your model, explore saved experiments, and test real predictions in a single workspace.</p><div className="hero-pills"><span><CircleDot size={12}/> Explainability</span><span><Zap size={12}/> Live inference</span><span><ShieldCheckIcon/> Local artifacts</span></div></div><div className="intelligence-art"><div className="art-orbit orbit-a"/><div className="art-orbit orbit-b"/><div className="art-core"><BrainCircuit size={31}/></div><span className="art-node node-a"><Activity size={15}/></span><span className="art-node node-b"><Target size={15}/></span><span className="art-node node-c"><TrendingUp size={15}/></span></div></div>
    <div className="intelligence-toolbar"><div><div className="toolbar-kicker">MODEL REGISTRY</div><h3>Your trained models</h3><p>Saved artifacts available for analysis and prediction.</p></div><button className="secondary-button refresh-models" onClick={() => void refresh()} disabled={loading}><RefreshCw size={14} className={loading ? 'spin' : ''}/> Refresh registry</button></div>
    {error && <div className="error-banner"><CircleDot size={16}/><span>{error}</span></div>}
    {loading ? <div className="intelligence-empty"><LoaderCircle className="spin" size={23}/><p>Finding saved models…</p></div> : models.length === 0 ? <div className="intelligence-empty"><div className="empty-brain"><Cpu size={25}/></div><h3>Your model registry is waiting</h3><p>Train a model in Model Studio first. April AI will save the winning pipeline here, ready to inspect and use.</p><button className="primary-button" onClick={() => document.getElementById('models')?.scrollIntoView({ behavior: 'smooth' })}>Go to Model Studio <ArrowRight size={15}/></button></div> : <>
      <div className="registry-layout"><div className="registry-list">{models.map((item, index) => <button className={`registry-item ${selected?.model_id === item.model_id ? 'selected' : ''}`} key={item.model_id} onClick={() => setSelectedId(item.model_id)}><span className={`registry-index index-${index % 3}`}>{String(index + 1).padStart(2, '0')}</span><span className="registry-main"><strong>{item.model}</strong><small>{item.task} · target: {item.target}</small><small>{prettyDate(item.created_at)}</small></span><span className="registry-arrow"><ArrowRight size={16}/></span></button>)}</div>
        {selected && <div className="registry-detail"><div className="registry-detail-top"><div><span className="model-status"><span/> SAVED ARTIFACT</span><h3>{selected.model}</h3><p>Predicting <strong>{selected.target}</strong> · {selected.task}</p></div><div className="model-emblem"><BrainCircuit size={23}/></div></div><div className="registry-stats"><div><span>FEATURES</span><strong>{selected.feature_count}</strong></div><div><span>ROWS USED</span><strong>{selected.rows_used.toLocaleString()}</strong></div><div><span>METRICS</span><strong>{metrics.length}</strong></div></div><div className="detail-metrics">{metrics.map(([key, value]) => <div className="detail-metric" key={key}><span>{key.replaceAll('_', ' ')}</span><strong>{value.toFixed(4)}</strong><div className="metric-meter"><i style={{ width: `${Math.max(2, Math.min(100, Math.abs(value) * 100))}%` }}/></div></div>)}</div><div className="artifact-id"><span>ARTIFACT ID</span><code>{selected.model_id}</code><button onClick={() => void navigator.clipboard?.writeText(selected.model_id)} aria-label="Copy model ID">Copy</button></div></div>}</div>
      {selected && <div className="intelligence-lower"><div className="panel intelligence-panel"><div className="panel-heading"><div><h3>What influences the model?</h3><p>Relative importance from the fitted estimator. This is a global signal, not causal proof.</p></div><Sparkles size={18} className="panel-icon"/></div>{selected.feature_importance.length ? <div className="importance-list">{selected.feature_importance.map((item, index) => <div className="importance-row" key={`${item.feature}-${index}`}><div className="importance-meta"><span title={item.feature}>{item.feature}</span><strong>{(item.importance * 100).toFixed(1)}%</strong></div><div className="importance-track"><i style={{ width: `${Math.max(1, item.importance * 100)}%` }}/></div></div>)}</div> : <div className="no-importance"><Sparkles size={18}/><p>This estimator doesn't expose built-in feature importance. Try a tree-based model to inspect feature influence.</p></div>}</div>
        <div className="panel prediction-panel"><div className="panel-heading"><div><h3>Prediction playground</h3><p>Enter values to run the saved pipeline.</p></div><Play size={18} className="panel-icon"/></div><div className="prediction-fields">{selected.feature_columns.map((feature) => <label className="prediction-field" key={feature}><span>{feature}<small>{selected.feature_types[feature] === 'number' ? 'NUMERIC' : 'TEXT / CATEGORY'}</small></span><input type={selected.feature_types[feature] === 'number' ? 'number' : 'text'} step={selected.feature_types[feature] === 'number' ? 'any' : undefined} value={values[feature] ?? ''} onChange={(event) => setValues((current) => ({ ...current, [feature]: event.target.value }))} placeholder={selected.feature_types[feature] === 'number' ? 'Enter a number' : 'Enter a category'}/></label>)}</div><button className="primary-button prediction-run" onClick={() => void runPrediction()} disabled={busy}>{busy ? <LoaderCircle className="spin" size={16}/> : <Play size={15}/>} {busy ? 'Running prediction…' : 'Generate prediction'} <ArrowRight size={15}/></button>{prediction && <div className="prediction-result"><span><CheckCircle2 size={14}/> PREDICTION RESULT</span><strong>{String(prediction.prediction)}</strong><small>Target: {prediction.target}</small>{prediction.probabilities && <div className="probability-list">{prediction.probabilities.map((item) => <div key={item.class}><span>{item.class}</span><div><i style={{ width: `${item.probability * 100}%` }}/></div><strong>{(item.probability * 100).toFixed(1)}%</strong></div>)}</div>}</div>}</div></div>}
    </>}
  </section>;
}
function ShieldCheckIcon() { return <CheckCircle2 size={12}/>; }

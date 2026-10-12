import { useState } from 'react';
import {
  Activity, BrainCircuit, CheckCircle2, LoaderCircle, Play,
  ShieldCheck, Trophy, TriangleAlert,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

type TrainingResult = {
  task: string;
  target: string;
  rows_used: number;
  training_rows: number;
  test_rows: number;
  feature_count: number;
  best_model: string;
  model_id: string;
  selection_metric: string;
  note: string;
  results: {
    model: string;
    status: string;
    error?: string;
    metrics: Record<string, unknown>;
  }[];
};

type Props = { file: File; columns: string[] };

function formatMetric(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—';
}

export default function TrainingStudio({ file, columns }: Props) {
  const [target, setTarget] = useState(columns[0] ?? '');
  const [task, setTask] = useState<'classification' | 'regression'>('classification');
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function train() {
    if (!target) { setError('Choose a target column first.'); return; }
    setBusy(true); setError(''); setResult(null);
    const body = new FormData();
    body.append('file', file);
    body.append('target', target);
    body.append('task', task);
    try {
      const response = await fetch(`${API_BASE}/api/models/train`, { method: 'POST', body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? 'Model training failed.');
      setResult(payload as TrainingResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unable to train models. Check the backend connection.');
    } finally { setBusy(false); }
  }

  return (
    <section className="training-studio" id="models">
      <div className="training-heading">
        <div>
          <div className="training-eyebrow"><BrainCircuit size={15} /> PHASE 04 · MODEL STUDIO</div>
          <h2>Turn data into <span>predictions.</span></h2>
          <p>Train baseline models on a held-out split, compare metrics, and save the best-performing pipeline.</p>
        </div>
        <div className="training-orb"><Activity size={27} /></div>
      </div>

      <div className="training-layout">
        <div className="panel training-controls">
          <div className="panel-heading"><div><h3>Training configuration</h3><p>Choose a target and prediction task.</p></div><BrainCircuit size={19} className="panel-icon" /></div>
          <div className="training-file"><div className="training-file-icon"><CheckCircle2 size={18} /></div><div><strong>{file.name}</strong><p>Dataset ready for model training</p></div></div>
          <label className="control-field training-target"><span>What do you want to predict?</span><select value={target} onChange={(event) => { setTarget(event.target.value); setResult(null); }}><option value="" disabled>Select target column</option>{columns.map((column) => <option value={column} key={column}>{column}</option>)}</select></label>
          <div className="task-picker">
            <button type="button" className={`task-option ${task === 'classification' ? 'selected' : ''}`} onClick={() => { setTask('classification'); setResult(null); }}><BrainCircuit size={20} /><strong>Classification</strong><span>Predict a category or class</span></button>
            <button type="button" className={`task-option ${task === 'regression' ? 'selected' : ''}`} onClick={() => { setTask('regression'); setResult(null); }}><Activity size={20} /><strong>Regression</strong><span>Predict a numeric value</span></button>
          </div>
          <div className="training-info"><ShieldCheck size={17} /><p>80/20 train/test split. Imputation, encoding, and scaling are fitted on training data only.</p></div>
          <button type="button" className="primary-button training-run" onClick={() => void train()} disabled={busy || columns.length < 2 || !target}>{busy ? <LoaderCircle className="spin" size={17} /> : <Play size={16} />}{busy ? 'Training models…' : 'Train and compare models'}</button>
          {columns.length < 2 && <p className="training-error">A dataset needs at least two columns: features and a target.</p>}
          {error && <div className="error-banner"><TriangleAlert size={17} /><span>{error}</span></div>}
        </div>
        <aside className="training-side-note">
          <div className="training-side-icon"><Trophy size={21} /></div><strong>Your model leaderboard</strong>
          <p>April AI compares three baseline algorithms for your selected task and saves the top-performing pipeline.</p>
          <div className="training-algorithm"><span>01</span><div><strong>{task === 'classification' ? 'Logistic Regression' : 'Linear Regression'}</strong><small>Interpretable baseline</small></div></div>
          <div className="training-algorithm"><span>02</span><div><strong>Decision Tree</strong><small>Captures nonlinear patterns</small></div></div>
          <div className="training-algorithm"><span>03</span><div><strong>Random Forest</strong><small>Ensemble baseline</small></div></div>
        </aside>
      </div>

      {result && <div className="training-results animate-in">
        <div className="section-title"><div><div className="success-label"><CheckCircle2 size={15} /> TRAINING COMPLETE</div><h2>Model leaderboard</h2><p>Target: <strong>{result.target}</strong> · {result.task} · {result.training_rows} training rows · {result.test_rows} test rows</p></div><span className="count-tag">{result.feature_count} features</span></div>
        <div className="training-summary-grid"><div className="comparison-card transformed"><span>BEST MODEL</span><strong className="best-model-name">{result.best_model}</strong><p>Ranked by {result.selection_metric === 'f1_weighted' ? 'weighted F1 score' : 'R² score'}</p></div><div className="comparison-card"><span>DATA USED</span><strong>{result.rows_used.toLocaleString()}</strong><p>Rows after target cleanup</p></div></div>
        <div className="panel leaderboard-panel"><div className="panel-heading"><div><h3>Performance comparison</h3><p>Metrics calculated on the held-out test set.</p></div><Trophy size={19} className="panel-icon" /></div>
          <div className="table-wrap"><table><thead><tr><th>Model</th><th>Status</th>{task === 'classification' ? <><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></> : <><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th></>}</tr></thead><tbody>{result.results.map((item) => { const metrics = item.metrics; return <tr key={item.model}><td className="column-name">{item.model}{item.model === result.best_model && <span className="winner-tag">BEST</span>}{item.error && <small className="training-error">{item.error}</small>}</td><td>{item.status === 'success' ? <span className="good-value">Complete</span> : <span className="missing-value">Failed</span>}</td>{task === 'classification' ? <><td>{formatMetric(metrics.accuracy)}</td><td>{formatMetric(metrics.precision_weighted)}</td><td>{formatMetric(metrics.recall_weighted)}</td><td>{formatMetric(metrics.f1_weighted)}</td></> : <><td>{formatMetric(metrics.mae)}</td><td>{formatMetric(metrics.rmse)}</td><td>{formatMetric(metrics.r2)}</td></>}</tr>; })}</tbody></table></div>
        </div>
        {task === 'classification' && <div className="panel leaderboard-panel"><div className="panel-heading"><div><h3>Confusion matrices</h3><p>Actual versus predicted classes for each trained model.</p></div></div><div className="confusion-grid">{result.results.filter((item) => item.status === 'success').map((item) => { const matrix = item.metrics.confusion_matrix as { labels: string[]; matrix: number[][] } | undefined; if (!matrix) return null; return <div className="confusion-card" key={item.model}><strong>{item.model}</strong><div className="preview-table-wrap"><table><thead><tr><th>Actual \ Predicted</th>{matrix.labels.map((label) => <th key={label}>{label}</th>)}</tr></thead><tbody>{matrix.matrix.map((row, index) => <tr key={matrix.labels[index]}><th>{matrix.labels[index]}</th>{row.map((value, j) => <td key={j}>{value}</td>)}</tr>)}</tbody></table></div></div>; })}</div></div>}
        <div className="leakage-note"><ShieldCheck size={18} /><p>{result.note}</p></div>
        <div className="training-model-id"><span>Saved model ID</span><code>{result.model_id}</code><button type="button" className="secondary-button" onClick={() => { void navigator.clipboard.writeText(result.model_id); }}>Copy ID</button></div>
      </div>}
    </section>
  );
}

import { ChangeEvent, DragEvent, ReactNode, useEffect, useRef, useState } from 'react';
import TrainingStudio from './TrainingStudio';
import ModelIntelligence from './ModelIntelligence';
import {
  Activity, ArrowUpRight, CheckCircle2, ChevronDown, CircleHelp, CloudUpload,
  Database, FileSpreadsheet, FlaskConical, Layers3, LoaderCircle, ShieldCheck,
  Sparkles, Table2, TriangleAlert, X, WandSparkles, SlidersHorizontal, RotateCcw,
} from 'lucide-react';

type ColumnProfile = {
  name: string;
  dtype: string;
  missing_count: number;
  missing_percent: number;
  unique_count: number;
  non_null_count: number;
};
type EdaBin = { label: string; count: number; start: number; end: number };
type EdaSummary = {
  numeric_summaries: { name: string; count: number; mean: number; std: number; min: number; q1: number; median: number; q3: number; max: number; skewness: number }[];
  distributions: { name: string; bins: EdaBin[] }[];
  outliers: { name: string; count: number; percent: number; lower_bound: number; upper_bound: number; method: string; sample_values: number[] }[];
  correlation: { columns: string[]; matrix: number[][] };
  categorical_distributions: { name: string; total_unique: number; categories: { label: string; count: number }[] }[];
  insights: { severity: string; title: string; detail: string }[];
  notes: string[];
};
type DatasetProfile = {
  filename: string;
  row_count: number;
  column_count: number;
  duplicate_row_count: number;
  total_missing_cells: number;
  numeric_column_count: number;
  categorical_column_count: number;
  numeric_columns: string[];
  categorical_columns: string[];
  columns: ColumnProfile[];
  preview: Record<string, unknown>[];
  preview_limit: number;
  eda: EdaSummary;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';
const formatNumber = (n: number) => new Intl.NumberFormat().format(n);
const displayValue = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value);

export default function App() {
  const fileInput = useRef<HTMLInputElement>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE}/api/health`);
      if (!response.ok) throw new Error('API unavailable');
      setApiStatus('online');
    } catch {
      setApiStatus('offline');
    }
  }

  // Check backend availability once when the page mounts.
  useEffect(() => { void checkHealth(); }, []);

  async function inspectFile(file?: File) {
    if (!file) return;
    setError('');
    setProfile(null);
    setUploadedFile(null);
    const allowed = /\.(csv|xlsx|xls|parquet)$/i.test(file.name);
    if (!allowed) {
      setError('Unsupported file type. Choose a CSV, XLSX, XLS, or Parquet file.');
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      setError('This file exceeds the 100 MB upload limit.');
      return;
    }
    setBusy(true);
    const body = new FormData();
    body.append('file', file);
    try {
      const response = await fetch(`${API_BASE}/api/datasets/inspect`, { method: 'POST', body });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? 'Unable to inspect this dataset.');
      setProfile(result as DatasetProfile);
      setUploadedFile(file);
      setApiStatus('online');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong while uploading.');
      setApiStatus('offline');
    } finally {
      setBusy(false);
    }
  }

  function onInputChange(event: ChangeEvent<HTMLInputElement>) {
    void inspectFile(event.target.files?.[0]);
    event.target.value = '';
  }
  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    void inspectFile(event.dataTransfer.files?.[0]);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><Activity size={21} /></div><div><strong>April <span>AI</span></strong><small>INTELLIGENT DATA WORKBENCH</small></div></div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list">
          <a className="nav-item active" href="#upload"><Database size={17} /> Dataset overview <span className="nav-dot" /></a>
          <a className="nav-item disabled" href="#eda"><Layers3 size={17} /> Exploratory analysis <span className="soon">LIVE</span></a>
          <a className="nav-item" href="#preprocessing"><FlaskConical size={17} /> Preprocessing <span className="soon">NEW</span></a>
          <a className="nav-item" href="#models"><Activity size={17} /> Model studio <span className="soon">04</span></a>
          <a className="nav-item" href="#intelligence"><Sparkles size={17} /> Intelligence <span className="soon">05</span></a>
        </nav>
        <div className="sidebar-bottom"><div className="status-card"><div className="status-icon"><ShieldCheck size={18} /></div><div><strong>Local workspace</strong><small>Your dataset stays in this session</small></div></div><div className="sidebar-foot"><span className={`status-led ${apiStatus}`} /> API {apiStatus === 'online' ? 'connected' : apiStatus === 'offline' ? 'unavailable' : 'checking'}<button className="icon-button" title="Check API connection" onClick={() => void checkHealth()}><ArrowUpRight size={15} /></button></div></div>
      </aside>

      <main className="main-content">
        <header className="topbar"><div className="breadcrumb">Workspace <span>/</span> <strong>Dataset overview</strong></div><div className="topbar-right"><div className="phase-pill"><span /> PHASE 05 · MODEL INTELLIGENCE</div><div className="avatar">AA</div></div></header>
        <section className="page-heading" id="upload"><div><div className="eyebrow"><Sparkles size={14} /> YOUR INTELLIGENT DATA WORKSPACE</div><h1>Make your data <span>work smarter.</span></h1><p>Start with a dataset. April AI will reveal patterns, quality issues, and relationships before you train a model.</p></div><div className="heading-decoration"><div className="deco-ring ring-one"/><div className="deco-ring ring-two"/><div className="deco-core"><Activity size={29}/></div></div></section>

        <section className="section-block"><div className="section-title"><div><h2>Dataset intake</h2><p>Upload a file to begin your analysis journey.</p></div><button className="help-button" title="Supported formats: CSV, Excel, Parquet"><CircleHelp size={17}/><span>Supported formats</span><ChevronDown size={14}/></button></div>
          <div className={`upload-zone ${dragging ? 'dragging' : ''} ${busy ? 'busy' : ''}`} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
            <input ref={fileInput} type="file" accept=".csv,.xlsx,.xls,.parquet" onChange={onInputChange} hidden />
            <div className="upload-icon">{busy ? <LoaderCircle className="spin" size={27}/> : <CloudUpload size={27}/>}</div>
            <h3>{busy ? 'Inspecting your dataset…' : 'Drop your dataset here'}</h3>
            <p>{busy ? 'Validating file and calculating data quality statistics.' : 'Drag and drop a file, or browse from your computer'}</p>
            <button className="primary-button" disabled={busy} onClick={() => fileInput.current?.click()}>{busy ? 'Please wait' : 'Browse files'} {!busy && <ArrowUpRight size={16}/>}</button>
            <div className="file-guidance"><span>CSV</span><span>Excel</span><span>Parquet</span><i/> Max 100 MB</div>
          </div>
          {error && <div className="error-banner"><TriangleAlert size={18}/><span>{error}</span><button className="icon-button" onClick={() => setError('')} aria-label="Dismiss error"><X size={16}/></button></div>}
        </section>

        {profile ? <DatasetResults profile={profile} file={uploadedFile} onReset={() => { setProfile(null); setUploadedFile(null); }} /> : <section className="section-block empty-state-block"><div className="section-title"><div><h2>Your analysis workspace</h2><p>Your dataset summary and data quality indicators will appear here.</p></div><span className="muted-tag">AWAITING DATA</span></div><div className="empty-grid"><div className="empty-card"><div className="empty-card-icon violet"><Table2 size={19}/></div><strong>Dataset profile</strong><p>Rows, columns, data types and missing values.</p><div className="skeleton-line wide"/><div className="skeleton-line short"/></div><div className="empty-card"><div className="empty-card-icon blue"><Activity size={19}/></div><strong>Data quality</strong><p>Duplicates and completeness checks.</p><div className="quality-placeholder"><span/><span/><span/><span/><span/><span/><span/></div></div><div className="empty-card"><div className="empty-card-icon amber"><Sparkles size={19}/></div><strong>Next best actions</strong><p>Guidance will be based on your actual dataset.</p><div className="skeleton-line wide"/><div className="skeleton-line medium"/></div></div></section>}
        <ModelIntelligence />
        <footer><span>April AI <b>·</b> Phase 5</span><span>Built for transparent, reproducible machine learning</span></footer>
      </main>
    </div>
  );
}

function DatasetResults({ profile, file, onReset }: { profile: DatasetProfile; file: File | null; onReset: () => void }) {
  const missingPercent = profile.row_count * profile.column_count ? (profile.total_missing_cells / (profile.row_count * profile.column_count)) * 100 : 0;
  return <section className="section-block results-section"><div className="section-title"><div><div className="success-label"><CheckCircle2 size={15}/> DATASET INSPECTED</div><h2>{profile.filename}</h2><p>Your automated exploratory data analysis is ready. Review distributions, outliers, correlations, and recommended checks below.</p></div><button className="secondary-button" onClick={onReset}><X size={15}/> Clear dataset</button></div>
    <div className="metrics-grid"><Metric label="Total rows" value={formatNumber(profile.row_count)} icon={<Table2 size={17}/>} note="Records detected"/><Metric label="Total columns" value={formatNumber(profile.column_count)} icon={<Database size={17}/>} note="Features detected"/><Metric label="Missing cells" value={formatNumber(profile.total_missing_cells)} icon={<TriangleAlert size={17}/>} note={`${missingPercent.toFixed(2)}% of all cells`} warn={profile.total_missing_cells > 0}/><Metric label="Duplicate rows" value={formatNumber(profile.duplicate_row_count)} icon={<Layers3 size={17}/>} note="Exact duplicate records" warn={profile.duplicate_row_count > 0}/></div>
    <div className="results-grid"><div className="panel column-panel"><div className="panel-heading"><div><h3>Column profile</h3><p>Detected types and completeness</p></div><span className="count-tag">{profile.columns.length} columns</span></div><div className="table-wrap"><table><thead><tr><th>Column</th><th>Data type</th><th>Missing</th><th>Unique</th></tr></thead><tbody>{profile.columns.map((column) => <tr key={column.name}><td className="column-name">{column.name}</td><td><span className="type-chip">{column.dtype}</span></td><td><span className={column.missing_count ? 'missing-value' : 'good-value'}>{column.missing_count ? `${column.missing_percent}%` : 'None'}</span></td><td>{formatNumber(column.unique_count)}</td></tr>)}</tbody></table></div><div className="panel-foot"><span><span className="legend-dot numeric"/> {profile.numeric_column_count} numeric</span><span><span className="legend-dot categorical"/> {profile.categorical_column_count} categorical / boolean</span></div></div>
    <div className="panel preview-panel"><div className="panel-heading"><div><h3>Data preview</h3><p>First {profile.preview_limit} rows at most</p></div><FileSpreadsheet size={19} className="panel-icon"/></div><div className="preview-table-wrap"><table><thead><tr>{profile.columns.map((column) => <th key={column.name}>{column.name}</th>)}</tr></thead><tbody>{profile.preview.map((row, index) => <tr key={index}>{profile.columns.map((column) => <td key={column.name}>{displayValue(row[column.name])}</td>)}</tr>)}</tbody></table></div><div className="preview-foot">Showing {profile.preview.length} of {formatNumber(profile.row_count)} rows</div></div></div>
    <EDASection eda={profile.eda} />
    {file && <PreprocessingStudio file={file} />}
    {file && <TrainingStudio file={file} columns={profile.columns.map((column) => column.name)} />}
  </section>;
}

function EDASection({ eda }: { eda: EdaSummary }) {
  const [selectedNumeric, setSelectedNumeric] = useState(eda.distributions[0]?.name ?? '');
  const selectedDistribution = eda.distributions.find((item) => item.name === selectedNumeric);
  const maxBin = Math.max(1, ...(selectedDistribution?.bins.map((bin) => bin.count) ?? []));
  const outlierTotal = eda.outliers.reduce((sum, item) => sum + item.count, 0);
  return <section className="eda-section" id="eda">
    <div className="section-title"><div><div className="success-label"><Sparkles size={15}/> AUTOMATED EDA</div><h2>Explore your data</h2><p>Statistical summaries and charts calculated from the uploaded dataset.</p></div><span className="count-tag">{eda.numeric_summaries.length} numeric · {eda.categorical_distributions.length} categorical</span></div>
    <div className="metrics-grid eda-metrics"><Metric label="Numeric features" value={formatNumber(eda.numeric_summaries.length)} icon={<Activity size={17}/>} note="Eligible for numeric analysis"/><Metric label="Potential outliers" value={formatNumber(outlierTotal)} icon={<TriangleAlert size={17}/>} note="IQR flags across numeric columns" warn={outlierTotal > 0}/><Metric label="Strong correlations" value={formatNumber(eda.correlation.matrix.reduce((sum, row, i) => sum + row.slice(i + 1).filter((v) => Math.abs(v) >= 0.8).length, 0))} icon={<Layers3 size={17}/>} note="Absolute Pearson r ≥ 0.80"/></div>
    <div className="eda-grid">
      <div className="panel chart-panel"><div className="panel-heading"><div><h3>Feature distributions</h3><p>Numeric histograms · counts per bin</p></div><select aria-label="Choose numeric feature" value={selectedNumeric} onChange={(e) => setSelectedNumeric(e.target.value)} disabled={!eda.distributions.length}>{eda.distributions.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}</select></div>
        {selectedDistribution ? <div className="histogram" role="img" aria-label={`Histogram for ${selectedNumeric}`}>{selectedDistribution.bins.map((bin, i) => <div className="histogram-column" key={`${bin.label}-${i}`} title={`${bin.label}: ${bin.count}`}><span className="histogram-count">{bin.count || ''}</span><div className="histogram-bar" style={{ height: `${Math.max(2, bin.count / maxBin * 100)}%` }}/><span className="histogram-label">{bin.label}</span></div>)}</div> : <p className="empty-chart">No numeric columns available for histograms.</p>}
      </div>
      <div className="panel chart-panel"><div className="panel-heading"><div><h3>Outlier scan</h3><p>IQR fences · review candidates, don't auto-delete</p></div><TriangleAlert size={19} className="panel-icon"/></div>
        {eda.outliers.length ? <div className="outlier-list">{eda.outliers.slice(0, 10).map((item) => <div className="outlier-row" key={item.name}><div className="outlier-name"><span>{item.name}</span><strong>{item.count} <small>({item.percent}%)</small></strong></div><div className="outlier-track"><div className={`outlier-fill ${item.count ? 'has-outliers' : ''}`} style={{ width: `${Math.min(100, item.percent)}%` }}/></div><div className="outlier-detail">Expected IQR range: {item.lower_bound} to {item.upper_bound}{item.sample_values.length ? ` · examples: ${item.sample_values.join(', ')}` : ''}</div></div>)}</div> : <p className="empty-chart">No numeric columns available for outlier checks.</p>}
      </div>
    </div>
    <div className="eda-grid">
      <div className="panel chart-panel"><div className="panel-heading"><div><h3>Correlation heatmap</h3><p>Pearson correlation · first 30 numeric features</p></div><Layers3 size={19} className="panel-icon"/></div>
        {eda.correlation.columns.length >= 2 ? <div className="heatmap-wrap"><table className="heatmap-table"><thead><tr><th></th>{eda.correlation.columns.map((col) => <th key={col} title={col}>{shortLabel(col)}</th>)}</tr></thead><tbody>{eda.correlation.matrix.map((row, i) => <tr key={eda.correlation.columns[i]}><th title={eda.correlation.columns[i]}>{shortLabel(eda.correlation.columns[i])}</th>{row.map((value, j) => { const strength = Math.abs(value); const background = value >= 0 ? `rgba(20, 184, 166, ${0.08 + strength * 0.72})` : `rgba(139, 92, 246, ${0.08 + strength * 0.72})`; return <td key={j} title={`${eda.correlation.columns[i]} × ${eda.correlation.columns[j]}: ${value.toFixed(2)}`} style={{ backgroundColor: background, color: strength > 0.55 ? '#ffffff' : 'var(--text)' }}>{value.toFixed(2)}</td>; })}</tr>)}</tbody></table><div className="heatmap-legend"><span>← Negative</span><span>Near zero</span><span>Positive →</span></div></div> : <p className="empty-chart">Correlation needs at least two numeric features.</p>}
      </div>
      <div className="panel chart-panel"><div className="panel-heading"><div><h3>Numeric summary</h3><p>Spread, central tendency and skewness</p></div><Database size={19} className="panel-icon"/></div>
        {eda.numeric_summaries.length ? <div className="table-wrap"><table><thead><tr><th>Feature</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th><th>Skew</th></tr></thead><tbody>{eda.numeric_summaries.map((item) => <tr key={item.name}><td className="column-name">{item.name}</td><td>{formatCompact(item.mean)}</td><td>{formatCompact(item.median)}</td><td>{formatCompact(item.min)}</td><td>{formatCompact(item.max)}</td><td><span className={Math.abs(item.skewness) >= 1 ? 'missing-value' : 'good-value'}>{item.skewness.toFixed(2)}</span></td></tr>)}</tbody></table></div> : <p className="empty-chart">No numeric columns found.</p>}
      </div>
    </div>
    {eda.categorical_distributions.length > 0 && <div className="panel chart-panel categorical-panel"><div className="panel-heading"><div><h3>Categorical breakdown</h3><p>Most frequent values in each categorical feature</p></div><Table2 size={19} className="panel-icon"/></div><div className="categorical-grid">{eda.categorical_distributions.slice(0, 6).map((item) => { const max = Math.max(1, ...item.categories.map((c) => c.count)); return <div className="category-card" key={item.name}><div className="category-title"><strong>{item.name}</strong><span>{item.total_unique} unique</span></div>{item.categories.slice(0, 6).map((category) => <div className="category-row" key={category.label}><span title={category.label}>{category.label}</span><div className="category-track"><div style={{ width: `${category.count / max * 100}%` }}/></div><strong>{category.count}</strong></div>)}</div>; })}</div></div>}
    <div className="panel insights-panel"><div className="panel-heading"><div><h3>Insights & recommendations</h3><p>Rule-based findings from the actual dataset</p></div><Sparkles size={19} className="panel-icon"/></div><div className="insights-list">{eda.insights.map((item, i) => <div className={`insight-item ${item.severity}`} key={`${item.title}-${i}`}><div className="insight-symbol">{item.severity === 'warning' ? <TriangleAlert size={17}/> : item.severity === 'success' ? <CheckCircle2 size={17}/> : <Sparkles size={17}/>}</div><div><strong className={item.title.startsWith('High cardinality:') ? 'high-cardinality-title' : ''}>{item.title}</strong><p>{item.detail}</p></div></div>)}</div><div className="eda-notes">{eda.notes.map((note) => <p key={note}>• {note}</p>)}</div></div>
  </section>;
}


type PreprocessResult = {
  original_shape: { rows: number; columns: number };
  transformed_shape: { rows: number; columns: number };
  rows_previewed: number;
  preview: Record<string, unknown>[];
  columns: { name: string; dtype: string; missing_count: number }[];
  actions: { step: string; detail: string }[];
  dropped_columns: string[];
  remaining_missing_cells: number;
  note: string;
};

function PreprocessingStudio({ file }: { file: File }) {
  const [missing, setMissing] = useState('auto');
  const [outliers, setOutliers] = useState('none');
  const [encoding, setEncoding] = useState('none');
  const [scaling, setScaling] = useState('none');
  const [engineering, setEngineering] = useState('none');
  const [selection, setSelection] = useState('none');
  const [threshold, setThreshold] = useState(0.9);
  const [constant, setConstant] = useState('0');
  const [result, setResult] = useState<PreprocessResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function runPreprocessing() {
    setBusy(true); setError(''); setResult(null);
    const body = new FormData();
    body.append('file', file);
    body.append('missing_strategy', missing);
    body.append('outlier_strategy', outliers);
    body.append('encoding_strategy', encoding);
    body.append('scaling_strategy', scaling);
    body.append('feature_engineering', engineering);
    body.append('feature_selection', selection);
    body.append('correlation_threshold', String(threshold));
    body.append('constant_value', constant);
    try {
      const response = await fetch(`${API_BASE}/api/datasets/preprocess`, { method: 'POST', body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? 'Preprocessing preview failed.');
      setResult(payload as PreprocessResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unable to preprocess this dataset.');
    } finally { setBusy(false); }
  }

  return <section className="preprocessing-studio" id="preprocessing">
    <div className="phase3-heading"><div><div className="success-label"><WandSparkles size={15}/> PHASE 03 · PREPROCESSING STUDIO</div><h2>Shape your data. <span>Keep control.</span></h2><p>Choose each transformation deliberately, then inspect the before-and-after preview before moving toward model training.</p></div><div className="phase3-orb"><SlidersHorizontal size={25}/></div></div>
    <div className="preprocess-layout">
      <div className="panel preprocess-controls"><div className="panel-heading"><div><h3>Transformation recipe</h3><p>Applied in order to a fresh copy of your uploaded file</p></div><SlidersHorizontal size={19} className="panel-icon"/></div>
        <div className="control-grid">
          <label className="control-field"><span>Missing values</span><select value={missing} onChange={e => setMissing(e.target.value)}><option value="auto">Auto · median / mode</option><option value="none">Leave as-is</option><option value="mean">Mean (numeric)</option><option value="median">Median (numeric)</option><option value="mode">Most frequent</option><option value="constant">Fill with constant</option><option value="drop_rows">Drop rows with missing values</option></select></label>
          {missing === 'constant' && <label className="control-field"><span>Constant fill value</span><input value={constant} onChange={e => setConstant(e.target.value)} placeholder="0 or unknown" /></label>}
          <label className="control-field"><span>Outlier handling</span><select value={outliers} onChange={e => setOutliers(e.target.value)}><option value="none">Don't change</option><option value="cap">Cap to IQR fences</option><option value="drop_rows">Remove rows with IQR outliers</option></select></label>
          <label className="control-field"><span>Categorical encoding</span><select value={encoding} onChange={e => setEncoding(e.target.value)}><option value="none">Don't encode</option><option value="onehot">One-hot encoding</option><option value="label">Label encoding</option></select></label>
          <label className="control-field"><span>Numeric scaling</span><select value={scaling} onChange={e => setScaling(e.target.value)}><option value="none">Don't scale</option><option value="standard">Standardize (z-score)</option><option value="minmax">Normalize (0–1)</option></select></label>
          <label className="control-field"><span>Feature engineering</span><select value={engineering} onChange={e => setEngineering(e.target.value)}><option value="none">No new features</option><option value="interactions">Numeric pairwise interactions</option></select></label>
          <label className="control-field"><span>Feature selection</span><select value={selection} onChange={e => setSelection(e.target.value)}><option value="none">Keep all features</option><option value="variance">Remove constant features</option><option value="correlation">Remove highly correlated features</option></select></label>
          {selection === 'correlation' && <label className="control-field"><span>Correlation threshold <strong>{threshold.toFixed(2)}</strong></span><input type="range" min="0.5" max="0.99" step="0.01" value={threshold} onChange={e => setThreshold(Number(e.target.value))}/></label>}
        </div>
        <div className="recipe-footer"><div className="recipe-file"><FileSpreadsheet size={16}/><span>{file.name}</span></div><button className="primary-button run-recipe" onClick={() => void runPreprocessing()} disabled={busy}>{busy ? <LoaderCircle size={16} className="spin"/> : <WandSparkles size={16}/>} {busy ? 'Applying recipe…' : 'Preview transformations'} <ArrowUpRight size={15}/></button></div>
        {error && <div className="error-banner"><TriangleAlert size={17}/><span>{error}</span></div>}
      </div>
      <div className="preprocess-side-note"><div className="side-note-icon"><ShieldCheck size={20}/></div><strong>Built for responsible ML</strong><p>April AI previews the impact before you commit. Outliers are not automatically bad data, and label codes don't imply meaningful order.</p><div className="mini-step"><span>01</span> Choose strategies</div><div className="mini-step"><span>02</span> Compare output</div><div className="mini-step"><span>03</span> Review before training</div></div>
    </div>
    {result && <div className="preprocess-results animate-in">
      <div className="section-title"><div><h3>Before & after</h3><p>Your transformation preview is ready. The original upload remains unchanged.</p></div><button className="secondary-button" onClick={() => setResult(null)}><RotateCcw size={14}/> Reset preview</button></div>
      <div className="comparison-grid"><div className="comparison-card"><span>ORIGINAL DATASET</span><strong>{formatNumber(result.original_shape.rows)} <small>rows</small></strong><p>{formatNumber(result.original_shape.columns)} columns</p></div><div className="comparison-arrow"><ArrowUpRight size={20}/></div><div className="comparison-card transformed"><span>TRANSFORMED PREVIEW</span><strong>{formatNumber(result.transformed_shape.rows)} <small>rows</small></strong><p>{formatNumber(result.transformed_shape.columns)} columns · {formatNumber(result.remaining_missing_cells)} missing cells remain</p></div></div>
      <div className="panel recipe-actions"><div className="panel-heading"><div><h3>What changed</h3><p>Traceable steps applied to this preview</p></div><CheckCircle2 size={19} className="panel-icon"/></div>{result.actions.length ? <div className="action-list">{result.actions.map((action, i) => <div className="action-row" key={`${action.step}-${i}`}><span>{String(i + 1).padStart(2, '0')}</span><div><strong>{action.step}</strong><p>{action.detail}</p></div></div>)}</div> : <p className="empty-chart">No transformation was needed for this recipe. Choose a strategy to change the preview.</p>}{result.dropped_columns.length > 0 && <p className="dropped-columns"><strong>Removed columns:</strong> {result.dropped_columns.join(', ')}</p>}</div>
      <div className="panel transformed-preview"><div className="panel-heading"><div><h3>Transformed data preview</h3><p>First {result.rows_previewed} rows · {result.columns.length} columns</p></div><Table2 size={18} className="panel-icon"/></div><div className="preview-table-wrap"><table><thead><tr>{result.columns.map(c => <th key={c.name}>{c.name}<small className="preview-dtype">{c.dtype}</small></th>)}</tr></thead><tbody>{result.preview.map((row, i) => <tr key={i}>{result.columns.map(c => <td key={c.name}>{displayValue(row[c.name])}</td>)}</tr>)}</tbody></table></div></div>
      <div className="leakage-note"><ShieldCheck size={18}/><p>{result.note}</p></div>
    </div>}
  </section>;
}

function shortLabel(value: string) { return value.length > 10 ? `${value.slice(0, 9)}…` : value; }
function formatCompact(value: number) { return new Intl.NumberFormat(undefined, { maximumFractionDigits: 3, notation: Math.abs(value) >= 10000 ? 'compact' : 'standard' }).format(value); }

function Metric({ label, value, icon, note, warn = false }: { label: string; value: string; icon: ReactNode; note: string; warn?: boolean }) {
  return <div className="metric-card"><div className={`metric-icon ${warn ? 'warn' : ''}`}>{icon}</div><div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className={`metric-note ${warn ? 'warn-text' : ''}`}>{note}</div></div>;
}

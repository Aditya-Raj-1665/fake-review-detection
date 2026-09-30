import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';

interface ModelInfo {
  model_name: string; model_version: string; demo: boolean;
  max_review_length: number; max_tokens: number | null;
}
interface Prediction {
  label: 'genuine' | 'deceptive'; authenticity_score: number; confidence: number;
  genuine_probability: number; deceptive_probability: number;
  model_name: string; model_version: string; demo: boolean; inference_ms: number;
  model_signals: { term: string; direction: string; weight: number }[];
}

const apiBase = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
const modelNames: Record<string, string> = {
  logistic_regression: 'Logistic Regression', naive_bayes: 'Naive Bayes', distilbert: 'DistilBERT',
};

/** Present one review, one probabilistic result, and clear model limitations. */
export default function App() {
  const [text, setText] = useState('');
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const maxLength = info?.max_review_length ?? 10000;

  useEffect(() => {
    const abort = new AbortController();
    fetch(`${apiBase}/api/v1/model/info`, { signal: abort.signal })
      .then(async response => {
        if (response.ok) setInfo(await response.json() as ModelInfo);
      }).catch(() => { /* Prediction requests show actionable connectivity errors. */ });
    return () => { abort.abort(); controller.current?.abort(); };
  }, []);

  async function analyze(event: FormEvent) {
    event.preventDefault();
    setError(''); setResult(null);
    if (!text.trim()) { setError('Enter a review to analyze.'); return; }
    if ([...text].length > maxLength) { setError(`Keep your review under ${maxLength} characters.`); return; }
    setLoading(true);
    const abort = new AbortController();
    controller.current = abort;
    const timer = window.setTimeout(() => abort.abort(), 60000);
    try {
      const response = await fetch(`${apiBase}/api/v1/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }), signal: abort.signal,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(typeof body?.detail === 'string' ? body.detail :
          'The review could not be analyzed. Please check your input and try again.');
      }
      setResult(await response.json() as Prediction);
    } catch (reason) {
      setError(reason instanceof Error && reason.name !== 'AbortError' &&
        reason.message !== 'Failed to fetch' ? reason.message :
        'Could not reach the model in time. Check that the backend is running and try again.');
    } finally { window.clearTimeout(timer); setLoading(false); }
  }

  return <main>
    <header className="masthead"><span className="mark" aria-hidden="true">ra</span>
      <span>REVIEW AUTHENTICITY</span><span className="project-tag">ML / NLP PROJECT</span>
    </header>
    <section className="intro">
      <p className="eyebrow">A SECOND LOOK AT THE WORDS</p>
      <h1>How authentic<br />does a review sound?</h1>
      <p className="lead">Explore the patterns a trained text classifier finds in a review.
        Get an estimated authenticity probability, with room for uncertainty.</p>
    </section>
    {info?.demo && <aside className="notice"><strong>Demo model active.</strong> Trained on a tiny
      synthetic dataset to demonstrate the pipeline. These predictions are not meaningful assessments.</aside>}
    <div className="workspace">
      <form onSubmit={analyze} className="panel input-panel">
        <div className="panel-heading"><span className="step">01</span><label htmlFor="review">Your review</label></div>
        <p className="hint">Paste an English product or service review.</p>
        <textarea id="review" placeholder="What did the reviewer say?" value={text}
          disabled={loading} aria-describedby="review-limit" onChange={event => {
            setText(event.target.value); setResult(null); setError('');
          }} />
        <div id="review-limit" className="input-meta"><span>Text only · no account needed</span>
          <span className={[...text].length > maxLength ? 'over-limit' : ''}>
            {[...text].length.toLocaleString()} / {maxLength.toLocaleString()}</span></div>
        {info?.max_tokens && <p className="hint">This model reads the first {info.max_tokens} tokens,
          including special tokens. The rest of a long review is not analyzed.</p>}
        <button disabled={loading || !text.trim() || [...text].length > maxLength} type="submit">
          {loading ? 'Analyzing…' : 'Analyze Review'}<span aria-hidden="true">↗</span></button>
        {error && <p role="alert" className="error">{error}</p>}
      </form>
      <section className="panel result-panel" aria-live="polite" aria-busy={loading}>
        <div className="panel-heading"><span className="step">02</span><h2>Model assessment</h2></div>
        {loading ? <div className="empty"><div className="loader" /><p>Reading the patterns in your review…</p></div>
          : result ? <div className="result">
            <p className={`label ${result.label}`}>Likely {result.label}</p>
            <p className="score">{result.authenticity_score.toFixed(1)}<span>%</span></p>
            <p className="score-caption">Estimated authenticity · P(genuine)</p>
            <meter min={0} max={100} value={result.authenticity_score}
              aria-label="Authenticity probability" />
            <div className="scale"><span>Deceptive</span><span>Genuine</span></div>
            <dl><div><dt>Confidence in chosen label</dt><dd>{(result.confidence * 100).toFixed(1)}%</dd></div>
              <div><dt>Model</dt><dd>{modelNames[result.model_name] ?? result.model_name}</dd></div>
              <div><dt>Version</dt><dd>{result.model_version}</dd></div>
              <div><dt>Model inference</dt><dd>{result.inference_ms.toFixed(1)} ms</dd></div></dl>
            {result.demo && <p className="notice">Demo output only. Do not use for real judgments.</p>}
            {result.model_signals.length > 0 && <details><summary>Model signals</summary>
              <p className="hint">Largest contributions to the underlying linear score.
                These are associations, not evidence of fraud.</p>
              <ul>{result.model_signals.map(signal => <li key={signal.term}>
                <span>“{signal.term}”</span><span>toward {signal.direction}</span></li>)}</ul></details>}
          </div> : <div className="empty"><span className="empty-icon" aria-hidden="true">≋</span>
            <h3>A probability, not a verdict.</h3><p>Your assessment will appear here after you analyze a review.</p></div>}
      </section>
    </div>
    <footer><span className="footnote-number">i</span><p>Predictions are probabilistic and should not be treated
      as definitive proof that a review is fraudulent. The Cornell benchmark concerns English hotel reviews;
      results may not transfer to other products, languages, or newer writing styles.</p></footer>
  </main>;
}

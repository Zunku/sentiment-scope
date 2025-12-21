import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
import nltk

#nltk.download('stopwords')
#nltk.download('rslp')
#nltk.download('punkt')
#nltk.download('wordnet')

__version__ = '0.1.0'
BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f'{BASE_DIR}/sentiment-analysis-pipeline-{__version__}.pkl', "rb") as f:
    pipeline = cast(Pipeline, pickle.load(f))


def _to_py(obj: Any) -> Any:
    """Convert numpy/scipy types to plain Python types for JSON serialisation."""
    try:
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return obj


def _explain_text_prep(text: str, text_prep: Any) -> List[Dict[str, Any]]:
    """Run each transformer in the text preprocessing pipeline and collect
    user-friendly summaries including input_text and output_text.
    """
    defs = {
        'regex': {
            'title': 'Regular Expressions (Regex)',
            'definition': 'Pattern-based text transformations used to detect or replace tokens like negations, numbers, dates, punctuation, and URLs.',
            'example': 'Negation: capture words like "no", "not", "never"; Numbers and Dates: mask digits; URLs: replace with <URL> token.'
        },
        'stopwords': {
            'title': 'Stop Words Removal',
            'definition': 'Elimination of common, uninformative words that do not add meaning to the sentiment.',
            'example': 'Removes words like "the", "and", "is" to keep informative tokens.'
        },
        'stemming': {
            'title': 'Stemming / Lemmatization',
            'definition': 'Reduce words to their root form so that related words are treated the same',
            'example': '"running", "runs" -> "run".'
        },
        'vectorizer': {
            'title': 'Vectorization (TF-IDF)',
            'definition': 'Convert tokens into numerical features. TF-IDF weights tokens by importance across documents.',
            'example': 'Token "great" may get higher weight if it appears often in positive examples and rarely overall.'
        }
    }

    steps: List[Dict[str, Any]] = []
    current: Any = [text]

    def _textify(x: Any) -> str:
        if x is None:
            return ''
        if isinstance(x, str):
            return x
        if isinstance(x, (list, tuple)):
            try:
                return ' '.join([str(i) for i in x])
            except Exception:
                return str(x)
        try:
            if hasattr(x, 'toarray'):
                arr = x.toarray()
                return str(arr)
            return str(x)
        except Exception:
            return repr(x)

    if hasattr(text_prep, 'steps'):
        for name, transformer in text_prep.steps:
            before = current
            try:
                out = transformer.transform(current)
            except Exception:
                try:
                    out = transformer.transform([text])
                except Exception:
                    out = None

            repr_out: Dict[str, Any] = {"type": type(out).__name__}
            try:
                if hasattr(out, 'toarray'):
                    arr = out.toarray()
                    flat = np.asarray(arr).ravel()
                    nonzero = int((flat != 0).sum())
                    repr_out.update({"shape": arr.shape, "nonzero": nonzero})
                    if hasattr(transformer, 'get_feature_names_out'):
                        fnames = list(transformer.get_feature_names_out())
                        # Use the actual transformed output (out) when available —
                        # calling transform([text]) can produce different inputs and
                        # may yield all-zero vectors. Fall back to transforming the
                        # original text only if out is None.
                        vec = out if out is not None else transformer.transform([text])
                        try:
                            if hasattr(vec, 'toarray'):
                                v = np.asarray(vec.toarray())[0]
                            else:
                                v = np.asarray(vec)[0]
                            indices = np.where(v != 0)[0]
                            # Provide more features for the UI (up to 200)
                            top = sorted([(fnames[i], float(v[i])) for i in indices], key=lambda x: -x[1])[:200]
                            repr_out['top_features'] = top
                        except Exception:
                            repr_out['top_features'] = []
                elif isinstance(out, (list, tuple)):
                    repr_out['preview'] = [_to_py(x) for x in (out[:200] if len(out) > 200 else out)]
                else:
                    repr_out['value'] = _to_py(out)
            except Exception:
                repr_out['value'] = repr(out)

            key = name.lower() if isinstance(name, str) else ''
            info = defs.get(key, None)
            block: Dict[str, Any] = {
                'step': name,
                'transformer': type(transformer).__name__,
                'title': info['title'] if info else (type(transformer).__name__),
                'definition': info['definition'] if info else '',
                'example': info['example'] if info else '',
                'input_text': _textify(before),
                'output_text': _textify(out),
                'result': repr_out,
            }
            steps.append(block)
            current = out
    else:
        name = getattr(text_prep, '__class__', type(text_prep)).__name__
        try:
            out = text_prep.transform([text])
        except Exception:
            out = None
        info = defs.get(name.lower(), None)
        steps.append({
            'step': name,
            'transformer': type(text_prep).__name__,
            'title': info['title'] if info else name,
            'definition': info['definition'] if info else '',
            'example': info['example'] if info else '',
            'input_text': text,
            'output_text': _textify(out),
            'result': _to_py(out)
        })

    return steps


def predict_sentiment(text: str) -> Dict[str, Any]:
    """Predict sentiment and return explainability details.

    Returns a dict with keys: `sentiment`, `probability` and `explain` (list of step summaries).
    """
    text_prep = pipeline.named_steps.get('text_prep')
    model = pipeline.named_steps.get('model')

    # Predict
    try:
        clean = text_prep.transform([text]) if text_prep is not None else [text]
    except Exception:
        clean = [text]

    pred = model.predict(clean)
    proba = model.predict_proba(clean)

    label = "Positive" if int(pred[0]) else "Negative"
    prob = float(proba[0][1] if int(pred[0]) else proba[0][0])

    # Build explain info (defensive, JSON-serialisable)
    explain: Optional[List[Dict[str, Any]]]
    try:
        explain = _explain_text_prep(text, text_prep) if text_prep is not None else None
    except Exception:
        explain = None

    # Try to compute SHAP contributions for features (best-effort).
    shap_info = None
    try:
        import shap

        vec = None
        if hasattr(text_prep, 'named_steps') and 'vectorizer' in text_prep.named_steps:
            vec = text_prep.named_steps['vectorizer']
        elif hasattr(text_prep, 'steps'):
            # attempt to find a vectorizer step
            for n, t in text_prep.steps:
                if 'vector' in n.lower() or 'tfidf' in type(t).__name__.lower():
                    vec = t
                    break

        if vec is not None and hasattr(vec, 'get_feature_names_out') and hasattr(model, 'coef_'):
            # convert to dense array for SHAP and model
            if hasattr(clean, 'toarray'):
                X = np.asarray(clean.toarray())
            else:
                X = np.asarray(clean)

            # LinearExplainer works well for linear models
            try:
                explainer = shap.LinearExplainer(model, X, feature_dependence="independent")
                shap_vals = explainer.shap_values(X)
            except Exception:
                # fallback to general Explainer
                explainer = shap.Explainer(model, X)
                shap_vals = explainer(X)

            # shap_vals may be array or object; try to extract numeric array for sample 0
            try:
                if hasattr(shap_vals, 'values'):
                    values = shap_vals.values
                else:
                    values = shap_vals
                # If values is list per class, pick class 1 (positive) if available
                if isinstance(values, list) and len(values) > 1:
                    arr = np.asarray(values[1])[0]
                else:
                    arr = np.asarray(values)[0]

                fnames = list(vec.get_feature_names_out())
                nonzero = np.where(X[0] != 0)[0]
                contribs = [{
                    'feature': fnames[i],
                    'value': float(arr[i])
                } for i in nonzero]

                # also produce token highlights (top contributors)
                token_highlights = sorted(contribs, key=lambda x: -abs(x['value']))[:30]
                shap_info = {'contributions': contribs, 'top': token_highlights}
            except Exception:
                shap_info = None
    except Exception:
        shap_info = None

    return {"sentiment": label, "probability": prob, "explain": explain, 'shap': shap_info}


# Example:
# print(predict_sentiment("Este produto é ótimo! Adorei a qualidade e o preço foi justo."))
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('predict');
  const text = document.getElementById('text');
  const result = document.getElementById('result');
  const sentiment = document.getElementById('sentiment');
  const bar = document.getElementById('bar');
  const probText = document.getElementById('probText');

  async function doPredict() {
    const value = text.value.trim();
    if (!value) return;
    btn.disabled = true;
    sentiment.textContent = '...' ;
    result.classList.remove('hidden');
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: value })
      });
      if (!res.ok) throw new Error('Prediction failed');
      const data = await res.json();
      sentiment.textContent = data.sentiment;
      const p = Math.round(data.probability * 100);
      bar.style.width = p + '%';
      probText.textContent = `Trust: ${p}%`;

      // Render explainability if present (styled)
      const explainPanel = document.getElementById('explainPanel');
      const explainDiv = document.getElementById('explain');
      const tokenBar = document.getElementById('tokenHighlights');
      explainDiv.innerHTML = '';
      tokenBar.innerHTML = '';
      if (data.explain && data.explain.length) {
        explainPanel.classList.remove('hidden');
        data.explain.forEach(block => {
          const container = document.createElement('div');
          container.className = 'explain-step card';

          const header = document.createElement('div');
          header.className = 'explain-header';
          const h = document.createElement('h4');
          h.textContent = block.title || `${block.step}`;
          header.appendChild(h);
          container.appendChild(header);
          const def = document.createElement('p');
          def.className = 'muted';
          def.textContent = block.definition || '';
          container.appendChild(def);

          if (block.example) {
            const ex = document.createElement('p');
            ex.innerHTML = '<strong>Example:</strong> ' + block.example;
            container.appendChild(ex);
          }

          // Show original and processed text (not JSON)
          const inp = document.createElement('p');
          inp.innerHTML = '<strong>Input:<br></strong> ' + (block.input_text || '');
          container.appendChild(inp);

          // For vectorizer steps we don't show the processed numeric matrix.
          const isVector = ((block.title || '') + ' ' + (block.transformer || '') + ' ' + (block.step || '')).toLowerCase().includes('vector');
          if (!isVector) {
            const outp = document.createElement('p');
            outp.innerHTML = '<strong>Processed:<br></strong> ' + (block.output_text || '');
            container.appendChild(outp);
          }

          // If vectorizer, show a clean list of important words and scores
          if (block.result && block.result.top_features && block.result.top_features.length) {
            const list = document.createElement('ol');
            list.className = 'feature-list';
            // show more features (up to 200)
            block.result.top_features.slice(0,200).forEach(([w, s]) => {
              const li = document.createElement('li');
              li.innerHTML = `<strong>${w}</strong>: ${Number(s).toFixed(6)}`;
              list.appendChild(li);
            });
            container.appendChild(document.createElement('hr'));
            const vs = document.createElement('div');
            vs.innerHTML = '<strong>Top features (word : score)</strong>';
            vs.appendChild(list);
            container.appendChild(vs);
          }

          explainDiv.appendChild(container);
        });
      } else {
        explainPanel.classList.add('hidden');
      }

      // Render SHAP token highlights and bar chart if present
      if (data.shap && data.shap.top && data.shap.top.length) {
        tokenBar.classList.remove('hidden');
        // token pills
        data.shap.top.slice(0,30).forEach(item => {
          const span = document.createElement('span');
          span.className = 'token';
          const weight = Math.tanh(item.value) ; // scale
          const color = weight > 0 ? `rgba(6,182,212,${Math.abs(weight)})` : `rgba(220,38,38,${Math.abs(weight)})`;
          span.style.background = color;
          span.textContent = item.feature;
          tokenBar.appendChild(span);
        });

        // SHAP bar chart area
        const shapContainer = document.createElement('div');
        shapContainer.className = 'shap-container';
        const title = document.createElement('h4');
        title.textContent = 'Top SHAP contributions';
        shapContainer.appendChild(title);
        // normalize absolute values for bar widths
        const vals = data.shap.top.map(x => Math.abs(x.value));
        const maxv = Math.max(...vals, 1e-6);
        data.shap.top.slice(0,20).forEach(item => {
          const row = document.createElement('div');
          row.className = 'shap-bar';
          const label = document.createElement('div');
          label.className = 'label';
          label.textContent = item.feature;
          const outer = document.createElement('div');
          outer.className = 'bar-outer';
          const inner = document.createElement('div');
          inner.className = 'bar-inner';
          const frac = Math.abs(item.value) / maxv;
          inner.style.width = Math.round(frac * 100) + '%';
          inner.style.background = item.value > 0 ? '#06b6d4' : '#dc2626';
          outer.appendChild(inner);
          const score = document.createElement('div');
          score.className = 'shap-score';
          score.textContent = Number(item.value).toFixed(4);
          row.appendChild(label);
          row.appendChild(outer);
          row.appendChild(score);
          shapContainer.appendChild(row);
        });
        explainDiv.insertBefore(shapContainer, explainDiv.firstChild);
      } else {
        tokenBar.classList.add('hidden');
      }
    } catch (err) {
      sentiment.textContent = 'Error';
      probText.textContent = String(err);
      bar.style.width = '0%';
    } finally {
      btn.disabled = false;
    }
  }

  btn.addEventListener('click', doPredict);
  text.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) doPredict();
  });
});

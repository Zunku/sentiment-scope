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

const form = document.getElementById('analysisForm');
const textArea = document.getElementById('newsText');
const charCount = document.getElementById('charCount');
const button = document.getElementById('analyzeButton');
const message = document.getElementById('formMessage');
const resultPanel = document.getElementById('resultPanel');
const emptyState = document.getElementById('emptyState');
const resultContent = document.getElementById('resultContent');

const examples = {
  real: "The district health department opened a vaccination centre at the government hospital on Monday. Officials published the eligibility rules, registration link and working hours in a signed notice on the department website.",
  fake: "SHOCKING secret cure removes every disease overnight! Doctors refuse to reveal it. Share this urgent message with ten groups before powerful companies delete the truth forever!"
};

textArea.addEventListener('input', () => {
  charCount.textContent = `${textArea.value.length.toLocaleString()} / 20,000`;
  message.textContent = '';
});

document.querySelectorAll('[data-example]').forEach((exampleButton) => {
  exampleButton.addEventListener('click', () => {
    textArea.value = examples[exampleButton.dataset.example];
    textArea.dispatchEvent(new Event('input'));
    textArea.focus();
  });
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = textArea.value.trim();
  if (text.length < 40) {
    message.textContent = 'Enter at least 40 characters for a meaningful analysis.';
    return;
  }

  button.disabled = true;
  button.querySelector('span').textContent = 'Analyzing with three models...';
  message.textContent = '';

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || 'Analysis failed.');
    renderResult(payload.result);
  } catch (error) {
    message.textContent = error.message || 'The classifier could not process this article.';
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Run ensemble analysis';
  }
});

function renderResult(result) {
  emptyState.hidden = true;
  resultContent.hidden = false;
  resultPanel.classList.remove('empty');

  const verdict = document.getElementById('verdictText');
  verdict.textContent = result.label;
  verdict.className = result.label.toLowerCase();
  document.getElementById('verdictExplanation').textContent = result.explanation;
  document.getElementById('agreementValue').textContent = `${result.agreement}%`;
  document.getElementById('agreementChart').style.setProperty('--score', `${result.agreement * 3.6}deg`);

  document.getElementById('voteGrid').innerHTML = Object.entries(result.votes).map(([model, vote]) => `
    <div class="vote-card"><span>${escapeHtml(model)}</span><strong class="${vote}">${vote}</strong></div>
  `).join('');

  const stats = [
    ['Words', result.stats.words],
    ['Sentences', result.stats.sentences],
    ['Characters', result.stats.characters],
    ['Avg. word length', result.stats.average_word_length]
  ];
  document.getElementById('statGrid').innerHTML = stats.map(([label, value]) => `
    <div><strong>${value}</strong><span>${label}</span></div>
  `).join('');

  document.getElementById('signalList').innerHTML = result.signals.map((signal) => `
    <div class="signal ${signal.type}">${escapeHtml(signal.text)}</div>
  `).join('');
  document.getElementById('disclaimer').textContent = result.disclaimer;
  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  })[char]);
}


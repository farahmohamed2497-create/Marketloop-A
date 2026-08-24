const state = { agentId: null, sessions: {} };
const agents = document.querySelector('#agents');
const messages = document.querySelector('#messages');
const title = document.querySelector('#agent-title');
const status = document.querySelector('#status');

function render(history) {
  messages.replaceChildren(...history.map(({role, content}) => {
    const item = document.createElement('p'); item.className = `message ${role}`;
    item.textContent = content; return item;
  })); messages.scrollTop = messages.scrollHeight;
}
async function selectAgent(agent) {
  state.agentId = agent.id; title.textContent = agent.name;
  status.textContent = agent.available ? 'Ready' : 'Integration pending';
  [...agents.children].forEach(button => button.classList.toggle('selected', button.dataset.id === agent.id));
  const session = state.sessions[agent.id];
  if (session) { const response = await fetch(`/api/chat/sessions/${session}`); render((await response.json()).history); }
  else render([]);
}
async function loadAgents() {
  const response = await fetch('/api/agents'); const payload = await response.json();
  payload.agents.forEach(agent => { const button = document.createElement('button'); button.textContent = agent.name;
    button.dataset.id = agent.id; button.disabled = !agent.available; button.onclick = () => selectAgent(agent); agents.append(button); });
  const first = payload.agents.find(agent => agent.available); if (first) selectAgent(first);
}
document.querySelector('#composer').addEventListener('submit', async event => {
  event.preventDefault(); if (!state.agentId) return;
  const input = document.querySelector('#message'); const message = input.value.trim(); if (!message) return;
  status.textContent = 'Thinking…'; input.disabled = true;
  try { const response = await fetch('/api/chat/messages', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({agent_id: state.agentId, session_id: state.sessions[state.agentId], message})});
    const payload = await response.json(); if (!response.ok) throw new Error(payload.error);
    state.sessions[state.agentId] = payload.session_id; render(payload.history); status.textContent = 'Ready'; input.value = '';
  } catch (error) { status.textContent = error.message; } finally { input.disabled = false; input.focus(); }
});
loadAgents();

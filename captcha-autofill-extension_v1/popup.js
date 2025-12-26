// popup.js
document.getElementById('solveBtn').addEventListener('click', async () => {
  const resultDiv = document.getElementById('result');
  const button = document.getElementById('solveBtn');
  
  // Show loading state
  button.textContent = 'Solving...';
  button.disabled = true;
  resultDiv.className = 'result';
  resultDiv.style.display = 'none';
  
  try {
    // Get active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Send message to content script
    chrome.tabs.sendMessage(tab.id, { action: 'solveCaptcha' }, (response) => {
      if (chrome.runtime.lastError) {
        resultDiv.className = 'result error';
        resultDiv.textContent = '❌ Error: ' + chrome.runtime.lastError.message;
        resultDiv.style.display = 'block';
      } else if (response && response.success) {
        resultDiv.className = 'result success';
        resultDiv.textContent = '✓ CAPTCHA solved successfully!';
        resultDiv.style.display = 'block';
      } else {
        resultDiv.className = 'result error';
        resultDiv.textContent = '❌ Failed to solve CAPTCHA';
        resultDiv.style.display = 'block';
      }
      
      button.textContent = 'Solve CAPTCHA Now';
      button.disabled = false;
    });
  } catch (error) {
    resultDiv.className = 'result error';
    resultDiv.textContent = '❌ Error: ' + error.message;
    resultDiv.style.display = 'block';
    button.textContent = 'Solve CAPTCHA Now';
    button.disabled = false;
  }
});

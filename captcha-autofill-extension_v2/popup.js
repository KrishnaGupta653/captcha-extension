// popup.js

// Initialize popup state
document.addEventListener('DOMContentLoaded', async () => {
  const toggleSwitch = document.getElementById('toggleSwitch');
  const statusDiv = document.getElementById('status');
  
  // Get current status
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]) {
      chrome.tabs.sendMessage(tabs[0].id, { action: 'getStatus' }, (response) => {
        if (response && response.success) {
          toggleSwitch.checked = response.enabled;
          updateStatus(response.enabled);
        }
      });
    }
  });
  
  // Toggle switch listener
  toggleSwitch.addEventListener('change', async () => {
    const enabled = toggleSwitch.checked;
    updateStatus(enabled);
    
    // Send message to content script
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { 
          action: 'toggleExtension', 
          enabled: enabled 
        }, (response) => {
          if (response && response.success) {
            console.log('Extension toggled:', enabled);
          }
        });
      }
    });
  });
  
  function updateStatus(enabled) {
    if (enabled) {
      statusDiv.className = 'status active';
      statusDiv.textContent = '✓ Extension is active and watching for CAPTCHAs';
    } else {
      statusDiv.className = 'status inactive';
      statusDiv.textContent = '✗ Extension is disabled';
    }
  }
});

// Solve button listener
document.getElementById('solveBtn').addEventListener('click', async () => {
  const resultDiv = document.getElementById('result');
  const button = document.getElementById('solveBtn');
  const toggleSwitch = document.getElementById('toggleSwitch');
  
  // Check if extension is enabled
  if (!toggleSwitch.checked) {
    resultDiv.className = 'result error';
    resultDiv.textContent = '✗ Please enable the extension first';
    resultDiv.style.display = 'block';
    setTimeout(() => {
      resultDiv.style.display = 'none';
    }, 3000);
    return;
  }
  
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
        resultDiv.textContent = '✗ Error: ' + chrome.runtime.lastError.message;
        resultDiv.style.display = 'block';
      } else if (response && response.success) {
        resultDiv.className = 'result success';
        resultDiv.textContent = '✓ ' + (response.message || 'CAPTCHA solved successfully!');
        resultDiv.style.display = 'block';
      } else {
        resultDiv.className = 'result error';
        resultDiv.textContent = '✗ ' + (response?.message || 'Failed to solve CAPTCHA');
        resultDiv.style.display = 'block';
      }
      
      button.textContent = 'Solve CAPTCHA Now';
      button.disabled = false;
      
      // Auto-hide result after 5 seconds
      setTimeout(() => {
        resultDiv.style.display = 'none';
      }, 5000);
    });
  } catch (error) {
    resultDiv.className = 'result error';
    resultDiv.textContent = '✗ Error: ' + error.message;
    resultDiv.style.display = 'block';
    button.textContent = 'Solve CAPTCHA Now';
    button.disabled = false;
  }
});
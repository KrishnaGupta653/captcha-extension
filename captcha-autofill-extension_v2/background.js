// background.js
chrome.runtime.onInstalled.addListener(() => {
  console.log('CAPTCHA Auto-Solver extension installed');
});

// Handle messages from popup or content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getStatus') {
    sendResponse({ status: 'active' });
  }
  return true;
});
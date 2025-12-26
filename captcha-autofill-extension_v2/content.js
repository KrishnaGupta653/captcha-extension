(async () => {
  console.log('[CAPTCHA Solver] Extension loaded');

  const CONFIG = {
    AUTO_SOLVE_ENABLED: true, // Toggle for auto-solving
    AUTO_CLICK_SUBMIT: false, // DISABLED - No auto-submit
    MIN_CHAR_DELAY: 50,       // Faster typing (was 100)
    MAX_CHAR_DELAY: 120,      // Fast typist speed (was 300)
    MOUSE_JITTER_MIN: 2,
    MOUSE_JITTER_MAX: 5,
  };

  let onnxSession = null;
  let vocab = null;

  // Check if extension is enabled
  async function isExtensionEnabled() {
    return new Promise((resolve) => {
      chrome.storage.local.get(['autoSolveEnabled'], (result) => {
        resolve(result.autoSolveEnabled !== false); // Default to true
      });
    });
  }

  // Set extension enabled state
  async function setExtensionEnabled(enabled) {
    return new Promise((resolve) => {
      chrome.storage.local.set({ autoSolveEnabled: enabled }, resolve);
    });
  }

  async function initialize() {
    try {
      if (onnxSession && vocab) return;

      const vocabUrl = chrome.runtime.getURL('vocab.json');
      const vocabResponse = await fetch(vocabUrl);
      vocab = await vocabResponse.json();

      const extensionUrl = chrome.runtime.getURL('');
      ort.env.wasm.wasmPaths = extensionUrl;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      ort.env.wasm.proxy = false;

      const modelUrl = chrome.runtime.getURL('captcha_solver.onnx');
      onnxSession = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });

      console.log('[CAPTCHA Solver] ✓ Model loaded');
      return true;
    } catch (error) {
      console.error('[CAPTCHA Solver] ✗ Init failed:', error);
      return false;
    }
  }

  function preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = 200;
    canvas.height = 64;
    ctx.drawImage(imageElement, 0, 0, 200, 64);
    
    const imageData = ctx.getImageData(0, 0, 200, 64);
    const data = imageData.data;
    const grayData = new Float32Array(1 * 1 * 64 * 200);
    
    for (let i = 0; i < 64; i++) {
      for (let j = 0; j < 200; j++) {
        const idx = (i * 200 + j) * 4;
        const gray = (data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114) / 255.0;
        grayData[i * 200 + j] = gray;
      }
    }
    
    return grayData;
  }

  function decodeCTC(output, vocab) {
    const [seqLen, batchSize, numClasses] = output.dims;
    const predictions = [];
    let prevClass = -1;
    
    for (let t = 0; t < seqLen; t++) {
      let maxIdx = 0;
      let maxVal = output.data[t * numClasses];
      
      for (let c = 1; c < numClasses; c++) {
        const val = output.data[t * numClasses + c];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = c;
        }
      }
      
      if (maxIdx !== 0 && maxIdx !== prevClass) {
        const char = vocab[maxIdx.toString()];
        if (char && char !== '<BLANK>') {
          predictions.push(char);
        }
      }
      
      prevClass = maxIdx;
    }
    
    return predictions.join('');
  }

  function getRandomDelay(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  /**
   * Human-like typing with mouse entropy
   * This function types text character by character with:
   * - Full event chain (keydown, keypress, input, keyup)
   * - Mouse jitter between keystrokes (simulates hand holding mouse)
   * - Fast typing like a skilled typist (50-120ms per char)
   * - NO SUBMISSION - stops after last character
   */
  async function typeHumanlyWithEntropy(inputElement, text) {
    console.log('[CAPTCHA Solver] Starting human-like typing:', text);
    
    // Get input field position for mouse movements
    const rect = inputElement.getBoundingClientRect();
    const baseX = rect.left + rect.width / 2;
    const baseY = rect.top + rect.height / 2;
    
    // Focus the input and clear it
    inputElement.focus();
    inputElement.value = '';
    
    // Type each character
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      const charCode = char.charCodeAt(0);
      const keyCode = char.toUpperCase().charCodeAt(0);
      
      // MOUSE ENTROPY: Simulate natural hand jitter
      const jitterX = getRandomDelay(-CONFIG.MOUSE_JITTER_MAX, CONFIG.MOUSE_JITTER_MAX);
      const jitterY = getRandomDelay(-CONFIG.MOUSE_JITTER_MAX, CONFIG.MOUSE_JITTER_MAX);
      
      try {
        const mouseMoveEvent = new MouseEvent('mousemove', {
          bubbles: true,
          cancelable: true,
          view: window,
          clientX: baseX + jitterX,
          clientY: baseY + jitterY,
          screenX: window.screenX + baseX + jitterX,
          screenY: window.screenY + baseY + jitterY,
          movementX: jitterX,
          movementY: jitterY
        });
        document.dispatchEvent(mouseMoveEvent);
      } catch (e) {
        // Mouse event failed, continue anyway
      }
      
      // Small delay after mouse move (faster typing)
      await new Promise(resolve => setTimeout(resolve, getRandomDelay(3, 8)));
      
      // EVENT CHAIN: Full standard keyboard event sequence
      
      // 1. keydown
      const keydownEvent = new KeyboardEvent('keydown', {
        bubbles: true,
        cancelable: true,
        key: char,
        code: `Key${char.toUpperCase()}`,
        keyCode: keyCode,
        charCode: 0,
        which: keyCode,
        view: window
      });
      inputElement.dispatchEvent(keydownEvent);
      
      // 2. keypress
      const keypressEvent = new KeyboardEvent('keypress', {
        bubbles: true,
        cancelable: true,
        key: char,
        code: `Key${char.toUpperCase()}`,
        keyCode: charCode,
        charCode: charCode,
        which: charCode,
        view: window
      });
      inputElement.dispatchEvent(keypressEvent);
      
      // 3. Update value and fire input event
      inputElement.value = inputElement.value + char;
      
      const inputEvent = new InputEvent('input', {
        bubbles: true,
        cancelable: true,
        data: char,
        inputType: 'insertText',
        view: window
      });
      inputElement.dispatchEvent(inputEvent);
      
      // 4. keyup
      const keyupEvent = new KeyboardEvent('keyup', {
        bubbles: true,
        cancelable: true,
        key: char,
        code: `Key${char.toUpperCase()}`,
        keyCode: keyCode,
        charCode: 0,
        which: keyCode,
        view: window
      });
      inputElement.dispatchEvent(keyupEvent);
      
      console.log(`[CAPTCHA Solver] Typed ${i + 1}/${text.length}: '${char}' (value: ${inputElement.value})`);
      
      // Fast typing - skilled typist speed (50-120ms per character)
      if (i < text.length - 1) {
        await new Promise(resolve => 
          setTimeout(resolve, getRandomDelay(CONFIG.MIN_CHAR_DELAY, CONFIG.MAX_CHAR_DELAY))
        );
      }
    }
    
    // Fire final change event
    const changeEvent = new Event('change', { bubbles: true });
    inputElement.dispatchEvent(changeEvent);
    
    // Visual feedback - green border
    const originalBorder = inputElement.style.border;
    inputElement.style.border = '2px solid #4CAF50';
    setTimeout(() => {
      inputElement.style.border = originalBorder;
    }, 2000);
    
    console.log('[CAPTCHA Solver] ✓ Typing complete. Final value:', inputElement.value);
  }

  async function solveCaptcha(imageElement) {
    try {
      await initialize();
      
      if (!onnxSession || !vocab) {
        throw new Error('Model not initialized');
      }

      const inputData = preprocessImage(imageElement);
      const tensor = new ort.Tensor('float32', inputData, [1, 1, 64, 200]);
      
      const feeds = { input: tensor };
      const results = await onnxSession.run(feeds);
      const output = results.output;
      
      const captchaText = decodeCTC(output, vocab);
      console.log('[CAPTCHA Solver] Predicted:', captchaText);
      
      return captchaText;
    } catch (error) {
      console.error('[CAPTCHA Solver] Error:', error);
      return null;
    }
  }

  async function findAndSolveCaptcha() {
    const enabled = await isExtensionEnabled();
    if (!enabled) {
      console.log('[CAPTCHA Solver] Extension is disabled');
      return { success: false, message: 'Extension is disabled' };
    }

    const captchaImages = document.querySelectorAll('img.captcha-img, img[alt*="Captcha"], img[alt*="captcha"]');
    
    if (captchaImages.length === 0) {
      console.log('[CAPTCHA Solver] No CAPTCHA found');
      return { success: false, message: 'No CAPTCHA found' };
    }

    for (const img of captchaImages) {
      if (!img.complete) {
        await new Promise(resolve => {
          img.onload = resolve;
          setTimeout(resolve, 2000);
        });
      }

      const captchaText = await solveCaptcha(img);
      
      if (captchaText) {
        const captchaInput = document.querySelector('input[name="captcha"], input#captcha, input[placeholder*="Captcha"], input[placeholder*="captcha"]');
        
        if (captchaInput) {
          // Clear existing value
          captchaInput.value = '';
          
          // Use human-like typing with mouse entropy
          await typeHumanlyWithEntropy(captchaInput, captchaText);
          
          console.log('[CAPTCHA Solver] ✓ Filled with human-like typing:', captchaText);
          
          // NO AUTO-SUBMIT - Execution stops here
          return { success: true, message: 'CAPTCHA solved successfully' };
        } else {
          console.warn('[CAPTCHA Solver] Input not found');
          return { success: false, message: 'Input field not found' };
        }
      }
    }
    
    return { success: false, message: 'Failed to solve CAPTCHA' };
  }

  // MutationObserver for automatic detection
  const observer = new MutationObserver(async (mutations) => {
    const enabled = await isExtensionEnabled();
    if (!enabled) return;

    for (const mutation of mutations) {
      if (mutation.addedNodes.length > 0) {
        for (const node of mutation.addedNodes) {
          if (node.nodeType === 1) {
            const hasCaptcha = node.querySelector && (
              node.querySelector('img.captcha-img') ||
              node.querySelector('img[alt*="Captcha"]') ||
              node.matches('img.captcha-img') ||
              node.matches('img[alt*="Captcha"]')
            );
            
            if (hasCaptcha) {
              console.log('[CAPTCHA Solver] New CAPTCHA detected');
              setTimeout(findAndSolveCaptcha, 500);
              break;
            }
          }
        }
      }
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

  // Initial check after page load
  setTimeout(async () => {
    const enabled = await isExtensionEnabled();
    if (enabled) {
      findAndSolveCaptcha();
    }
  }, 1000);

  // Message listener for popup commands
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'solveCaptcha') {
      findAndSolveCaptcha().then((result) => {
        sendResponse(result);
      }).catch(error => {
        sendResponse({ success: false, error: error.message });
      });
      return true; // Keep channel open for async response
    }
    
    if (request.action === 'toggleExtension') {
      setExtensionEnabled(request.enabled).then(() => {
        sendResponse({ success: true, enabled: request.enabled });
      });
      return true;
    }
    
    if (request.action === 'getStatus') {
      isExtensionEnabled().then((enabled) => {
        sendResponse({ success: true, enabled: enabled });
      });
      return true;
    }
  });

  console.log('[CAPTCHA Solver] Ready with human-like typing');
})();
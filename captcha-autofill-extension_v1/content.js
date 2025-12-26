(async () => {
  console.log('[CAPTCHA Solver] Extension loaded');

  const CONFIG = {
    AUTO_CLICK_SUBMIT: true,
    SUBMIT_DELAY_MIN: 1500,
    SUBMIT_DELAY_MAX: 3500,
  };

  let onnxSession = null;
  let vocab = null;

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
      console.error('[CAPTCHA Solver] ❌ Init failed:', error);
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
    const captchaImages = document.querySelectorAll('img.captcha-img, img[alt*="Captcha"], img[alt*="captcha"]');
    
    if (captchaImages.length === 0) {
      console.log('[CAPTCHA Solver] No CAPTCHA found');
      return;
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
          captchaInput.value = captchaText;
          captchaInput.dispatchEvent(new Event('input', { bubbles: true }));
          captchaInput.dispatchEvent(new Event('change', { bubbles: true }));
          
          console.log('[CAPTCHA Solver] ✓ Filled:', captchaText);

          const originalBorder = captchaInput.style.border;
          captchaInput.style.border = '2px solid #4CAF50';
          setTimeout(() => {
            captchaInput.style.border = originalBorder;
          }, 2000);

          if (CONFIG.AUTO_CLICK_SUBMIT) {
            const submitButtons = document.querySelectorAll(
              'button[type="submit"].search_btn.train_Search_custom_hover, ' +
              'button[type="submit"].train_Search, ' +
              'button[type="submit"].btnDefault.train_Search, ' +
              'button.search_btn, ' +
              'button.train_Search, ' +
              'button[type="submit"]'
            );

            if (submitButtons.length > 0) {
              const delay = getRandomDelay(CONFIG.SUBMIT_DELAY_MIN, CONFIG.SUBMIT_DELAY_MAX);
              console.log(`[CAPTCHA Solver] Clicking submit in ${delay}ms`);
              
              setTimeout(() => {
                const button = submitButtons[0];
                button.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
                button.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
                button.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
                button.click();
                console.log('[CAPTCHA Solver] ✓ Clicked:', button.textContent.trim());
              }, delay);
            }
          }
        } else {
          console.warn('[CAPTCHA Solver] Input not found');
        }
      }
    }
  }

  const observer = new MutationObserver((mutations) => {
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

  setTimeout(findAndSolveCaptcha, 1000);

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'solveCaptcha') {
      findAndSolveCaptcha().then(() => {
        sendResponse({ success: true });
      }).catch(error => {
        sendResponse({ success: false, error: error.message });
      });
      return true;
    }
  });

  console.log('[CAPTCHA Solver] Ready');
})();
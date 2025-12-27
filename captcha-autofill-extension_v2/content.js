(async () => {
  console.log("[CAPTCHA Solver] Loaded");

  // ==========================================
  // FIXED: Added missing CONFIG and helper
  // ==========================================
  const CONFIG = {
    MIN_CHAR_DELAY: 80,
    MAX_CHAR_DELAY: 150,
    MOUSE_JITTER_MAX: 5,
  };

  function getRandomDelay(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  let onnxSession = null;
  let vocab = null;

  // Initialize model
  async function initialize() {
    if (onnxSession && vocab) return true;

    try {
      const vocabUrl = chrome.runtime.getURL("vocab.json");
      vocab = await (await fetch(vocabUrl)).json();

      const extensionUrl = chrome.runtime.getURL("");
      ort.env.wasm.wasmPaths = extensionUrl;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;

      const modelUrl = chrome.runtime.getURL("captcha_solver.onnx");
      onnxSession = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });

      console.log("[CAPTCHA Solver] Model ready");
      return true;
    } catch (error) {
      console.error("[CAPTCHA Solver] Init error:", error);
      return false;
    }
  }

  // Preprocess image
  function preprocessImage(img) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = 200;
    canvas.height = 64;
    ctx.drawImage(img, 0, 0, 200, 64);

    const imageData = ctx.getImageData(0, 0, 200, 64);
    const data = imageData.data;
    const grayData = new Float32Array(64 * 200);

    for (let i = 0; i < 64; i++) {
      for (let j = 0; j < 200; j++) {
        const idx = (i * 200 + j) * 4;
        const gray =
          (data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114) /
          255.0;
        grayData[i * 200 + j] = gray;
      }
    }

    return grayData;
  }

  // Decode CTC output
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
        if (char && char !== "<BLANK>") {
          predictions.push(char);
        }
      }
      prevClass = maxIdx;
    }

    return predictions.join("");
  }

  // ==========================================
  // FIXED: Consistent random delays + robust typing
  // ==========================================
  async function fillCaptcha(input, text) {
    console.log("[CAPTCHA Solver] Typing:", text);

    const rect = input.getBoundingClientRect();
    const baseX = rect.left + rect.width / 2;
    const baseY = rect.top + rect.height / 2;

    // Get native setter (bypass Angular/React)
    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      "value"
    ).set;

    // Clear and focus
    input.value = "";
    input.focus();
    input.dispatchEvent(new FocusEvent("focus", { bubbles: true }));
    input.dispatchEvent(new Event("click", { bubbles: true }));

    await new Promise((resolve) => setTimeout(resolve, 50));

    // Type each character
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      const charCode = char.charCodeAt(0);
      const keyCode = char.toUpperCase().charCodeAt(0);

      // Mouse jitter (simulate hand movement)
      const jitterX = getRandomDelay(
        -CONFIG.MOUSE_JITTER_MAX,
        CONFIG.MOUSE_JITTER_MAX
      );
      const jitterY = getRandomDelay(
        -CONFIG.MOUSE_JITTER_MAX,
        CONFIG.MOUSE_JITTER_MAX
      );

      try {
        document.dispatchEvent(
          new MouseEvent("mousemove", {
            bubbles: true,
            clientX: baseX + jitterX,
            clientY: baseY + jitterY,
          })
        );
      } catch (e) {}

      await new Promise((resolve) => setTimeout(resolve, getRandomDelay(3, 8)));

      // Store expected value
      const expectedValue = input.value + char;

      // 1. Keydown event
      input.dispatchEvent(
        new KeyboardEvent("keydown", {
          bubbles: true,
          cancelable: true,
          key: char,
          code: `Key${char.toUpperCase()}`,
          keyCode: keyCode,
          which: keyCode,
        })
      );

      // 2. Set value using native setter (critical for Angular/React)
      nativeInputValueSetter.call(input, expectedValue);

      // 3. Input event
      input.dispatchEvent(
        new InputEvent("input", {
          bubbles: true,
          cancelable: false,
          data: char,
          inputType: "insertText",
        })
      );

      // 4. Keyup event
      input.dispatchEvent(
        new KeyboardEvent("keyup", {
          bubbles: true,
          cancelable: true,
          key: char,
          code: `Key${char.toUpperCase()}`,
          keyCode: keyCode,
          which: keyCode,
        })
      );

      // Verify value
      if (input.value !== expectedValue) {
        console.warn(
          "[CAPTCHA Solver] Value mismatch, forcing:",
          expectedValue
        );
        nativeInputValueSetter.call(input, expectedValue);
      }

      // Random delay between keystrokes
      if (i < text.length - 1) {
        await new Promise((resolve) =>
          setTimeout(
            resolve,
            getRandomDelay(CONFIG.MIN_CHAR_DELAY, CONFIG.MAX_CHAR_DELAY)
          )
        );
      }
    }

    // Final events
    input.dispatchEvent(new Event("change", { bubbles: true }));
    input.dispatchEvent(new Event("blur", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 10));
    input.dispatchEvent(new FocusEvent("focus", { bubbles: true }));

    // Visual feedback
    const border = input.style.border;
    input.style.border = "2px solid #4CAF50";
    setTimeout(() => (input.style.border = border), 2000);

    console.log("[CAPTCHA Solver] Complete:", input.value);
  }

  // Main solve function
  async function solveCaptcha() {
    try {
      await initialize();
      if (!onnxSession || !vocab) return false;

      // Find CAPTCHA image
      const img = document.querySelector(
        'img.captcha-img, img[alt*="Captcha"], img[alt*="captcha"]'
      );
      if (!img) {
        console.log("[CAPTCHA Solver] No image found");
        return false;
      }

      // Wait for image to load
      if (!img.complete) {
        await new Promise((resolve) => {
          img.onload = resolve;
          setTimeout(resolve, 2000);
        });
      }

      // Solve
      const inputData = preprocessImage(img);
      const tensor = new ort.Tensor("float32", inputData, [1, 1, 64, 200]);
      const results = await onnxSession.run({ input: tensor });
      const captchaText = decodeCTC(results.output, vocab);

      if (!captchaText) {
        console.log("[CAPTCHA Solver] No text decoded");
        return false;
      }

      console.log("[CAPTCHA Solver] Solved:", captchaText);

      // Find input field
      const input = document.querySelector(
        'input[name="captcha"], input#captcha, ' +
          'input[placeholder*="Captcha"], input[placeholder*="captcha"]'
      );

      if (!input) {
        console.log("[CAPTCHA Solver] No input found");
        return false;
      }

      // Fill it
      await fillCaptcha(input, captchaText);
      return true;
    } catch (error) {
      console.error("[CAPTCHA Solver] Error:", error);
      return false;
    }
  }

  // Auto-detect CAPTCHA on page load
  setTimeout(() => {
    const img = document.querySelector(
      'img.captcha-img, img[alt*="Captcha"], img[alt*="captcha"]'
    );
    if (img) {
      console.log("[CAPTCHA Solver] Auto-solving...");
      solveCaptcha();
    }
  }, 1000);

  // Watch for new CAPTCHAs
  let solving = false;
  const observer = new MutationObserver(() => {
    if (solving) return;

    const img = document.querySelector(
      'img.captcha-img, img[alt*="Captcha"], img[alt*="captcha"]'
    );
    const input = document.querySelector(
      'input[name="captcha"], input#captcha, ' +
        'input[placeholder*="Captcha"], input[placeholder*="captcha"]'
    );

    if (img && input && !input.value) {
      console.log("[CAPTCHA Solver] New CAPTCHA detected");
      solving = true;
      solveCaptcha().finally(() => {
        setTimeout(() => (solving = false), 1000);
      });
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });

  // Manual trigger from popup
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "solveCaptcha") {
      solveCaptcha().then((success) => {
        sendResponse({
          success: success,
          message: success ? "CAPTCHA solved" : "Failed to solve",
        });
      });
      return true;
    }
  });

  console.log("[CAPTCHA Solver] Ready");
})();

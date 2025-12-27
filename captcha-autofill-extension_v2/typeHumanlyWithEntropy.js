/**
 * Types text into an input field with human-like behavior including:
 * - Natural keystroke timing variations
 * - Mouse movement entropy between keystrokes
 * - Full event chain for each character
 *
 * @param {HTMLInputElement} inputElement - The target input field
 * @param {string} text - The text to type (will be typed exactly, no typos)
 * @returns {Promise<void>} - Resolves when typing is complete
 */
async function typeHumanlyWithEntropy(inputElement, text) {
  console.log("[CAPTCHA Solver] Starting human-like typing:", text);

  // Get input field position for mouse movements
  const rect = inputElement.getBoundingClientRect();
  const baseX = rect.left + rect.width / 2;
  const baseY = rect.top + rect.height / 2;

  // Focus the input and clear it
  inputElement.focus();
  inputElement.value = "";

  // Trigger initial focus event
  inputElement.dispatchEvent(new FocusEvent("focus", { bubbles: true }));

  // Type each character
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const charCode = char.charCodeAt(0);
    const keyCode = char.toUpperCase().charCodeAt(0);

    // MOUSE ENTROPY: Simulate natural hand jitter
    const jitterX = getRandomDelay(
      -CONFIG.MOUSE_JITTER_MAX,
      CONFIG.MOUSE_JITTER_MAX
    );
    const jitterY = getRandomDelay(
      -CONFIG.MOUSE_JITTER_MAX,
      CONFIG.MOUSE_JITTER_MAX
    );

    try {
      const mouseMoveEvent = new MouseEvent("mousemove", {
        bubbles: true,
        cancelable: true,
        view: window,
        clientX: baseX + jitterX,
        clientY: baseY + jitterY,
        screenX: window.screenX + baseX + jitterX,
        screenY: window.screenY + baseY + jitterY,
        movementX: jitterX,
        movementY: jitterY,
      });
      document.dispatchEvent(mouseMoveEvent);
    } catch (e) {
      // Mouse event failed, continue anyway
    }

    // Small delay after mouse move (faster typing)
    await new Promise((resolve) => setTimeout(resolve, getRandomDelay(3, 8)));

    // Store the current value before modification
    const previousValue = inputElement.value;
    const newValue = previousValue + char;

    // EVENT CHAIN: Simplified event sequence to avoid duplication

    // 1. keydown
    const keydownEvent = new KeyboardEvent("keydown", {
      bubbles: true,
      cancelable: true,
      key: char,
      code: `Key${char.toUpperCase()}`,
      keyCode: keyCode,
      charCode: 0,
      which: keyCode,
      view: window,
    });
    inputElement.dispatchEvent(keydownEvent);

    // 2. Update value directly (only once!)
    inputElement.value = newValue;

    // 3. Fire input event with the character
    const inputEvent = new InputEvent("input", {
      bubbles: true,
      cancelable: false,
      data: char,
      inputType: "insertText",
      view: window,
    });
    inputElement.dispatchEvent(inputEvent);

    // 4. keyup
    const keyupEvent = new KeyboardEvent("keyup", {
      bubbles: true,
      cancelable: true,
      key: char,
      code: `Key${char.toUpperCase()}`,
      keyCode: keyCode,
      charCode: 0,
      which: keyCode,
      view: window,
    });
    inputElement.dispatchEvent(keyupEvent);

    // Verify the value is correct
    if (inputElement.value !== newValue) {
      console.warn(
        `[CAPTCHA Solver] Value mismatch! Expected: ${newValue}, Got: ${inputElement.value}`
      );
      inputElement.value = newValue; // Force correct value
    }

    console.log(
      `[CAPTCHA Solver] Typed ${i + 1}/${text.length}: '${char}' (value: ${
        inputElement.value
      })`
    );

    // Fast typing - skilled typist speed (50-120ms per character)
    if (i < text.length - 1) {
      await new Promise((resolve) =>
        setTimeout(
          resolve,
          getRandomDelay(CONFIG.MIN_CHAR_DELAY, CONFIG.MAX_CHAR_DELAY)
        )
      );
    }
  }

  // Fire final change event
  const changeEvent = new Event("change", { bubbles: true });
  inputElement.dispatchEvent(changeEvent);

  // Trigger blur and focus back for Angular detection
  inputElement.dispatchEvent(new Event("blur", { bubbles: true }));
  await new Promise((resolve) => setTimeout(resolve, 10));
  inputElement.dispatchEvent(new FocusEvent("focus", { bubbles: true }));

  // Visual feedback - green border
  const originalBorder = inputElement.style.border;
  inputElement.style.border = "2px solid #4CAF50";
  setTimeout(() => {
    inputElement.style.border = originalBorder;
  }, 2000);

  console.log(
    "[CAPTCHA Solver] ✓ Typing complete. Final value:",
    inputElement.value
  );
}

// Example usage in your content.js:
// const captchaInput = document.querySelector('input[name="captcha"]');
// const captchaText = 'ABC123';
// await typeHumanlyWithEntropy(captchaInput, captchaText);

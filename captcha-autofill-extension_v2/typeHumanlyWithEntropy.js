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
  // Ensure the input element is focused
  inputElement.focus();
  
  // Get the bounding rectangle of the input element for mouse positioning
  const rect = inputElement.getBoundingClientRect();
  const baseX = rect.left + rect.width / 2;
  const baseY = rect.top + rect.height / 2;
  
  /**
   * Generates a random delay between min and max milliseconds
   */
  const getRandomDelay = (min = 100, max = 300) => {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  };
  
  /**
   * Generates a small random offset for mouse jitter
   */
  const getMouseJitter = (range = 5) => {
    return Math.floor(Math.random() * (range * 2 + 1)) - range;
  };
  
  /**
   * Dispatches a mouse move event with slight jitter
   */
  const dispatchMouseJitter = () => {
    const jitterX = getMouseJitter(3);
    const jitterY = getMouseJitter(3);
    
    const mouseMoveEvent = new MouseEvent('mousemove', {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: baseX + jitterX,
      clientY: baseY + jitterY,
      screenX: window.screenX + baseX + jitterX,
      screenY: window.screenY + baseY + jitterY
    });
    
    inputElement.dispatchEvent(mouseMoveEvent);
  };
  
  /**
   * Dispatches the full event chain for a single character
   */
  const typeCharacter = (char) => {
    const charCode = char.charCodeAt(0);
    const keyCode = char.toUpperCase().charCodeAt(0);
    
    // 1. keydown event
    const keydownEvent = new KeyboardEvent('keydown', {
      bubbles: true,
      cancelable: true,
      key: char,
      code: `Key${char.toUpperCase()}`,
      charCode: 0,
      keyCode: keyCode,
      which: keyCode
    });
    inputElement.dispatchEvent(keydownEvent);
    
    // 2. keypress event (deprecated but some sites still use it)
    const keypressEvent = new KeyboardEvent('keypress', {
      bubbles: true,
      cancelable: true,
      key: char,
      charCode: charCode,
      keyCode: charCode,
      which: charCode
    });
    inputElement.dispatchEvent(keypressEvent);
    
    // 3. Update the actual input value
    const currentValue = inputElement.value;
    inputElement.value = currentValue + char;
    
    // 4. textInput event (for some browsers)
    if (typeof TextEvent !== 'undefined') {
      const textInputEvent = new TextEvent('textInput', {
        bubbles: true,
        cancelable: true,
        data: char
      });
      inputElement.dispatchEvent(textInputEvent);
    }
    
    // 5. input event (modern standard)
    const inputEvent = new Event('input', {
      bubbles: true,
      cancelable: true
    });
    inputElement.dispatchEvent(inputEvent);
    
    // 6. keyup event
    const keyupEvent = new KeyboardEvent('keyup', {
      bubbles: true,
      cancelable: true,
      key: char,
      code: `Key${char.toUpperCase()}`,
      charCode: 0,
      keyCode: keyCode,
      which: keyCode
    });
    inputElement.dispatchEvent(keyupEvent);
  };
  
  // Type each character with delays and mouse jitter
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    
    // Add mouse jitter before typing
    dispatchMouseJitter();
    
    // Small delay for mouse movement to register
    await new Promise(resolve => setTimeout(resolve, 20));
    
    // Type the character
    typeCharacter(char);
    
    // Add another mouse jitter after typing
    dispatchMouseJitter();
    
    // Wait before next character (except for the last one)
    if (i < text.length - 1) {
      const delay = getRandomDelay(100, 300);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  
  // Final change event to ensure the input is properly registered
  const changeEvent = new Event('change', {
    bubbles: true,
    cancelable: true
  });
  inputElement.dispatchEvent(changeEvent);
  
  // Final mouse jitter
  dispatchMouseJitter();
  
  console.log('[Human Typing] Completed typing:', text);
}

// Example usage in your content.js:
// const captchaInput = document.querySelector('input[name="captcha"]');
// const captchaText = 'ABC123';
// await typeHumanlyWithEntropy(captchaInput, captchaText);
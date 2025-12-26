// ==========================================
// CAPTCHA SOLVER - FAST MODE
// ==========================================

const CAPTCHA_SERVER_URL = "http://localhost:5000/solve";
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function solveCaptcha() {
  try {
    console.log("🚀 Starting fast CAPTCHA solve...");
    
    // Find and extract CAPTCHA image
    const captchaImg = document.querySelector('img.captcha-img[alt="Captcha Image here"]');
    if (!captchaImg) throw new Error("❌ CAPTCHA image not found");
    
    const imageData = captchaImg.src;
    if (!imageData || !imageData.startsWith('data:image')) throw new Error("❌ Invalid image data");

    // Send to solver
    const response = await fetch(CAPTCHA_SERVER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });

    if (!response.ok) throw new Error(`❌ Server error: ${response.status}`);
    
    const result = await response.json();
    if (!result.success) throw new Error(`❌ Solver error: ${result.error}`);

    console.log("✓ CAPTCHA SOLVED:", result.text);
    
    // Fill CAPTCHA
    const captchaInput = document.querySelector('input#captcha[formcontrolname="captcha"]');
    if (!captchaInput) throw new Error("❌ Input not found");
    
    captchaInput.focus();
    captchaInput.value = result.text;
    captchaInput.dispatchEvent(new Event('input', { bubbles: true }));
    captchaInput.dispatchEvent(new Event('change', { bubbles: true }));
    
    await wait(300);

    // Click Continue
    let continueButton = document.querySelector('button[type="submit"].train_Search.btnDefault');
    if (!continueButton) {
      const buttons = document.querySelectorAll("button.train_Search.btnDefault");
      for (let button of buttons) {
        if (button.textContent.trim().includes("Continue")) {
          continueButton = button;
          break;
        }
      }
    }
    if (!continueButton) throw new Error("❌ Continue button not found");
    
    continueButton.click();
    console.log("✓ Continue clicked");
    
    // Wait for Pay & Book button
    await wait(500);
    let payBookButton = null;
    let retries = 0;
    
    while (!payBookButton && retries < 15) {
      payBookButton = document.querySelector('button.btn.btn-primary.hidden-xs.ng-star-inserted');
      if (!payBookButton) {
        await wait(150);
        retries++;
      }
    }
    
    if (!payBookButton) throw new Error("❌ Pay & Book not found");
    
    payBookButton.click();
    console.log("✓ Pay & Book clicked");
    
    // Wait for QR payment link
    await wait(500);
    let qrPaymentSpan = null;
    let qrRetries = 0;
    
    while (!qrPaymentSpan && qrRetries < 15) {
      qrPaymentSpan = document.querySelector('span[onclick="submitUpiQrForm()"]') || document.querySelector('#PayByQrButton span');
      if (!qrPaymentSpan) {
        await wait(150);
        qrRetries++;
      }
    }
    
    if (!qrPaymentSpan) throw new Error("❌ QR payment not found");
    
    qrPaymentSpan.click();
    console.log("✓ QR payment clicked");
    
    console.log("✅ COMPLETE! All steps done");
    return result.text;

  } catch (error) {
    console.error("❌ FAILED:", error.message);
    throw error;
  }
}

// Auto-run
console.log("🎯 IRCTC FAST CAPTCHA SOLVER");
console.log("Starting in 1 second...\n");

setTimeout(() => {
  solveCaptcha().catch(err => console.error("Error:", err.message));
}, 1000);

window.testCaptcha = solveCaptcha;
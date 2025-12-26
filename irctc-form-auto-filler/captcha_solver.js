// // ==========================================
// // CAPTCHA SOLVER TEST - Standalone Script
// // ==========================================

// const CAPTCHA_SERVER_URL = "http://localhost:5000/solve";

// const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// async function solveCaptcha() {
//   try {
//     console.log("=== Starting CAPTCHA Solver Test ===");
//     console.log("Step 1: Finding CAPTCHA image...");
    
//     // Find the CAPTCHA image
//     const captchaImg = document.querySelector('img.captcha-img[alt="Captcha Image here"]');
//     if (!captchaImg) {
//       throw new Error("❌ CAPTCHA image not found");
//     }
//     console.log("✓ CAPTCHA image found");

//     // Get the base64 image data
//     const imageData = captchaImg.src;
//     if (!imageData || !imageData.startsWith('data:image')) {
//       throw new Error("❌ Invalid CAPTCHA image data");
//     }
//     console.log("✓ Image data extracted:", imageData.substring(0, 50) + "...");

//     console.log("\nStep 2: Sending CAPTCHA to solver server...");
//     console.log("Server URL:", CAPTCHA_SERVER_URL);

//     // Send to Flask server
//     const response = await fetch(CAPTCHA_SERVER_URL, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({
//         image: imageData
//       })
//     });

//     if (!response.ok) {
//       throw new Error(`❌ Server responded with status: ${response.status}`);
//     }
//     console.log("✓ Server responded successfully");

//     const result = await response.json();
//     console.log("Server response:", result);

//     if (!result.success) {
//       throw new Error(`❌ CAPTCHA solver error: ${result.error}`);
//     }

//     console.log("\n✓✓✓ CAPTCHA SOLVED: '" + result.text + "' ✓✓✓\n");
    
//     // Step 3: Fill in the CAPTCHA text
//     console.log("Step 3: Finding CAPTCHA input field...");
//     const captchaInput = document.querySelector('input#captcha[formcontrolname="captcha"]');
//     if (!captchaInput) {
//       throw new Error("❌ CAPTCHA input field not found");
//     }
//     console.log("✓ CAPTCHA input field found");

//     // Trigger input events to register the value
//     console.log("Step 4: Filling CAPTCHA text...");
//     captchaInput.focus();
//     captchaInput.value = result.text;
//     captchaInput.dispatchEvent(new Event('input', { bubbles: true }));
//     captchaInput.dispatchEvent(new Event('change', { bubbles: true }));
//     captchaInput.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));
    
//     console.log("✓ CAPTCHA text filled:", result.text);
//     await wait(500);

//     // Step 4: Click Continue button
//     console.log("\nStep 5: Finding Continue button...");
//     let continueButton = document.querySelector('button[type="submit"].train_Search.btnDefault');

//     if (!continueButton) {
//       const buttons = document.querySelectorAll("button.train_Search.btnDefault");
//       for (let button of buttons) {
//         if (button.textContent.trim().includes("Continue")) {
//           continueButton = button;
//           break;
//         }
//       }
//     }

//     if (!continueButton) {
//       throw new Error("❌ Continue button not found");
//     }
//     console.log("✓ Continue button found");

//     console.log("\nStep 6: Clicking Continue button...");
//     continueButton.scrollIntoView({ behavior: "smooth", block: "center" });
//     await wait(300);
//     continueButton.click();
    
//     console.log("\n" + "=".repeat(50));
//     console.log("✓✓✓ SUCCESS! CAPTCHA SOLVED AND SUBMITTED ✓✓✓");
//     console.log("=".repeat(50));

//     return result.text;

//   } catch (error) {
//     console.error("\n" + "=".repeat(50));
//     console.error("❌ CAPTCHA SOLVING FAILED");
//     console.error("=".repeat(50));
//     console.error("Error:", error.message);
//     console.error("\nTroubleshooting:");
//     console.error("1. Is Flask server running? Check: curl http://localhost:5000/health");
//     console.error("2. Is model file present? Check: crnn_captcha_model.pth");
//     console.error("3. Are you on the passenger details page with CAPTCHA visible?");
//     console.error("4. Check browser console for CORS errors");
//     throw error;
//   }
// }

// // ==========================================
// // RUN TEST
// // ==========================================

// console.log("\n" + "=".repeat(60));
// console.log("IRCTC CAPTCHA SOLVER - STANDALONE TEST");
// console.log("=".repeat(60));
// console.log("\nMake sure:");
// console.log("1. You are on the passenger details page");
// console.log("2. CAPTCHA is visible on the page");
// console.log("3. Flask server is running (python crnn_server.py)");
// console.log("\nStarting test in 2 seconds...\n");

// setTimeout(() => {
//   solveCaptcha().catch(err => {
//     console.error("\nTest failed:", err.message);
//   });
// }, 2000);

// // You can also call it manually with:
// // solveCaptcha();
// window.testCaptcha = solveCaptcha;
// ==========================================
// CAPTCHA SOLVER TEST - Standalone Script
// ==========================================

const CAPTCHA_SERVER_URL = "http://localhost:5000/solve";

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function solveCaptcha() {
  try {
    console.log("=== Starting CAPTCHA Solver Test ===");
    console.log("Step 1: Finding CAPTCHA image...");
    
    // Find the CAPTCHA image
    const captchaImg = document.querySelector('img.captcha-img[alt="Captcha Image here"]');
    if (!captchaImg) {
      throw new Error("❌ CAPTCHA image not found");
    }
    console.log("✓ CAPTCHA image found");

    // Get the base64 image data
    const imageData = captchaImg.src;
    if (!imageData || !imageData.startsWith('data:image')) {
      throw new Error("❌ Invalid CAPTCHA image data");
    }
    console.log("✓ Image data extracted:", imageData.substring(0, 50) + "...");

    console.log("\nStep 2: Sending CAPTCHA to solver server...");
    console.log("Server URL:", CAPTCHA_SERVER_URL);

    // Send to Flask server
    const response = await fetch(CAPTCHA_SERVER_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: imageData
      })
    });

    if (!response.ok) {
      throw new Error(`❌ Server responded with status: ${response.status}`);
    }
    console.log("✓ Server responded successfully");

    const result = await response.json();
    console.log("Server response:", result);

    if (!result.success) {
      throw new Error(`❌ CAPTCHA solver error: ${result.error}`);
    }

    console.log("\n✓✓✓ CAPTCHA SOLVED: '" + result.text + "' ✓✓✓\n");
    
    // Step 3: Fill in the CAPTCHA text
    console.log("Step 3: Finding CAPTCHA input field...");
    const captchaInput = document.querySelector('input#captcha[formcontrolname="captcha"]');
    if (!captchaInput) {
      throw new Error("❌ CAPTCHA input field not found");
    }
    console.log("✓ CAPTCHA input field found");

    // Trigger input events to register the value
    console.log("Step 4: Filling CAPTCHA text...");
    captchaInput.focus();
    captchaInput.value = result.text;
    captchaInput.dispatchEvent(new Event('input', { bubbles: true }));
    captchaInput.dispatchEvent(new Event('change', { bubbles: true }));
    captchaInput.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));
    
    console.log("✓ CAPTCHA text filled:", result.text);
    await wait(500);

    // Step 5: Click Continue button
    console.log("\nStep 5: Finding Continue button...");
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

    if (!continueButton) {
      throw new Error("❌ Continue button not found");
    }
    console.log("✓ Continue button found");

    console.log("\nStep 6: Clicking Continue button...");
    continueButton.scrollIntoView({ behavior: "smooth", block: "center" });
    await wait(300);
    continueButton.click();
    
    console.log("✓ Continue button clicked");
    console.log("\n=== Waiting for next page to load ===");
    
    // Wait for page to load
    await wait(3000);

    // Step 7: Click Pay & Book button
    console.log("\nStep 7: Finding Pay & Book button...");
    const payBookButton = document.querySelector('button.btn.btn-primary.hidden-xs.ng-star-inserted');
    
    if (!payBookButton) {
      throw new Error("❌ Pay & Book button not found");
    }
    console.log("✓ Pay & Book button found");

    console.log("\nStep 8: Clicking Pay & Book button...");
    payBookButton.scrollIntoView({ behavior: "smooth", block: "center" });
    await wait(300);
    payBookButton.click();
    
    console.log("✓ Pay & Book button clicked");
    console.log("\n=== Waiting for payment page to load ===");
    
    // Wait for payment page to load
    await wait(3000);

    // Step 9: Click QR payment link
    console.log("\nStep 9: Finding QR payment link...");
    const qrPaymentSpan = document.querySelector('span[onclick="submitUpiQrForm()"]');
    if (!qrPaymentSpan) {
      qrPaymentSpan = document.querySelector('#PayByQrButton span');
    }
    if (!qrPaymentSpan) {
      throw new Error("❌ QR payment link not found");
    }
    console.log("✓ QR payment link found");

    console.log("\nStep 10: Clicking QR payment link...");
    qrPaymentSpan.scrollIntoView({ behavior: "smooth", block: "center" });
    await wait(300);
    qrPaymentSpan.click();
    
    console.log("✓ QR payment link clicked");
    
    console.log("\n" + "=".repeat(50));
    console.log("✓✓✓ SUCCESS! COMPLETE FLOW EXECUTED ✓✓✓");
    console.log("=".repeat(50));
    console.log("\nSteps completed:");
    console.log("1. CAPTCHA solved and submitted");
    console.log("2. Pay & Book button clicked");
    console.log("3. QR payment option selected");

    return result.text;

  } catch (error) {
    console.error("\n" + "=".repeat(50));
    console.error("❌ PROCESS FAILED");
    console.error("=".repeat(50));
    console.error("Error:", error.message);
    console.error("\nTroubleshooting:");
    console.error("1. Is Flask server running? Check: curl http://localhost:5000/health");
    console.error("2. Is model file present? Check: crnn_captcha_model.pth");
    console.error("3. Are you on the passenger details page with CAPTCHA visible?");
    console.error("4. Check browser console for CORS errors");
    console.error("5. Verify page transitions are complete before next click");
    throw error;
  }
}

// ==========================================
// RUN TEST
// ==========================================

console.log("\n" + "=".repeat(60));
console.log("IRCTC CAPTCHA SOLVER WITH PAYMENT FLOW");
console.log("=".repeat(60));
console.log("\nMake sure:");
console.log("1. You are on the passenger details page");
console.log("2. CAPTCHA is visible on the page");
console.log("3. Flask server is running (python crnn_server.py)");
console.log("\nStarting test in 2 seconds...\n");

setTimeout(() => {
  solveCaptcha().catch(err => {
    console.error("\nTest failed:", err.message);
  });
}, 2000);

// You can also call it manually with:
// solveCaptcha();
window.testCaptcha = solveCaptcha;
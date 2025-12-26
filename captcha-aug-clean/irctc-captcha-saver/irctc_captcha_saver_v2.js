async function downloadCaptchas(count, delay = 2000) {
  for (let i = 0; i < count; i++) {
    try {
      // 1. Find the Captcha Image - check both possible locations
      let img = document.querySelector(".captcha-img");
      
      // Check if image exists and is visible
      if (!img || img.offsetParent === null) {
        // Try alternative selector if first one not found or not visible
        img = document.querySelector(".captcha_div img.captcha-img");
      }
      
      if (!img) {
        console.error("Captcha image not found!");
        break;
      }

      // Check if image is actually visible on screen
      const isVisible = img.offsetParent !== null && 
                       window.getComputedStyle(img).display !== 'none' &&
                       window.getComputedStyle(img).visibility !== 'hidden';
      
      if (!isVisible) {
        console.warn("Captcha image found but not visible. Skipping.");
        break;
      }

      // 2. Create a canvas to convert the image to a PNG
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      const ctx = canvas.getContext("2d");
      
      // Handle both regular src and base64 data URLs
      ctx.drawImage(img, 0, 0);

      // 3. Convert to Data URL and Trigger Download
      const dataUrl = canvas.toDataURL("image/png");
      const link = document.createElement("a");
      link.href = dataUrl;
      link.download = `captcha_${Date.now()}_${i + 1}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      console.log(`Downloaded ${i + 1}/${count}`);

      // Wait a moment for download to initiate
      await new Promise((resolve) => setTimeout(resolve, 500));

      // 4. Refresh the captcha by dispatching events instead of direct click
      let refreshBtn = document.querySelector(".captcha_div a.pull-right");
      
      if (!refreshBtn) {
        // Try alternative selector
        refreshBtn = document.querySelector("a[aria-label*='refresh']");
      }
      
      if (!refreshBtn) {
        // Try finding by glyphicon class
        refreshBtn = document.querySelector(".glyphicon-repeat")?.closest("a");
      }
      
      if (refreshBtn) {
        // Use MouseEvent instead of .click() to better simulate user interaction
        const clickEvent = new MouseEvent('click', {
          view: window,
          bubbles: true,
          cancelable: true,
          button: 0
        });
        refreshBtn.dispatchEvent(clickEvent);
        
        // Alternative: trigger Angular event if it's an Angular app
        // refreshBtn.dispatchEvent(new Event('click', { bubbles: true }));
      } else {
        console.warn("Refresh button not found. Stopping.");
        break;
      }

      // 5. Wait for the new image to load before next loop
      console.log(`Waiting ${delay}ms before next captcha...`);
      await new Promise((resolve) => setTimeout(resolve, delay));
      
    } catch (err) {
      console.error(`Error at step ${i + 1}:`, err);
      // Continue to next iteration instead of breaking
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  console.log("Download process completed!");
}

// Start downloading 20 captchas with a 2-second gap
downloadCaptchas(20, 2000);
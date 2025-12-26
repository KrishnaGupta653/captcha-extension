async function downloadCaptchas(count, delay = 1000) {
  for (let i = 0; i < count; i++) {
    try {
      // 1. Find the Captcha Image
      const img = document.querySelector(".captcha-img");
      if (!img) {
        console.error("Captcha image not found!");
        break;
      }

      // 2. Create a canvas to convert the image to a PNG
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      // 3. Convert to Data URL and Trigger Download
      const dataUrl = canvas.toDataURL("image/png");
      const link = document.createElement("a");
      link.href = dataUrl;
      link.download = `captcha_${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      console.log(`Downloaded ${i + 1}/${count}`);

      // 4. Click the Refresh Button
      const refreshBtn = document.querySelector("a.pull-right");
      if (refreshBtn) {
        refreshBtn.click();
      } else {
        console.warn("Refresh button not found. Stopping.");
        break;
      }

      // 5. Wait for the new image to load before next loop
      await new Promise((resolve) => setTimeout(resolve, delay));
    } catch (err) {
      console.error("Error at step " + i, err);
    }
  }
}

// Start downloading 20 captchas with a 2-second gap
downloadCaptchas(20);

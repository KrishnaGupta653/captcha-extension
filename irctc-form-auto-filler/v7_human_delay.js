// ==========================================
// IRCTC HUMAN-LIKE AUTOFILL SCRIPT
// ==========================================

const TICKET_CONFIG = {
    FROM_STATION_CODE: "KPD",
  FROM_STATION_FULL: "KATPADI JN - KPD (VELLORE)",
  TO_STATION_CODE: "RNC",
  TO_STATION_FULL: "RANCHI - RNC (HATIA/RANCHI)",
  TRAVEL_DATE: "22/12/2025",
  TRAIN_CLASS: "All Classes",
  QUOTA: "TATKAL",
  TARGET_TRAIN_NUMBER: "13352",
  PREFERRED_CLASS: "SL",
  ALTERNATIVE_CLASSES: ["SL", "3E", "2A", "1A"],
  PASSENGERS: [
//   FROM_STATION_CODE: "GZB",
//   FROM_STATION_FULL: "GHAZIABAD - GZB",
//   TO_STATION_CODE: "HRI",
//   TO_STATION_FULL: "HARDOI - HRI",
//   TRAVEL_DATE: "22/12/2025",
//   TRAIN_CLASS: "All Classes",
//   QUOTA: "PREMIUM TATKAL",
//   TARGET_TRAIN_NUMBER: "12230",
//   PREFERRED_CLASS: "3A",
//   ALTERNATIVE_CLASSES: ["SL", "3E", "2A", "1A"],
//   PASSENGERS: [
    {
      name: "Ayush Choudhary",
      age: 21,
      gender: "M",
      berthPreference: "UB",
      nationality: "IN",
    },
  ],
  AUTO_UPGRADATION: true,
  CONFIRM_BERTHS_ONLY: true,
  PAYMENT_TYPE: "UPI",
  UPI_ID: "yourname@paytm",
  
  // Human-like timing control (in milliseconds)
  TYPING_SPEED: 8,        // Time between keystrokes (50-150 recommended)
  ACTION_DELAY: 500,       // Delay between major actions (400-1000 recommended)
  CLICK_DELAY: 200,        // Time before clicking (200-500 recommended)
};

const RETRY_CONFIG = {
  SEARCH_BUTTON_INTERVAL: 400,
  BOOK_NOW_INTERVAL: 400,
  MAX_RETRY_DURATION: 120000,
  ERROR_DIALOG_CHECK_INTERVAL: 200,
};

let shouldStopRetrying = false;
let errorDialogInterval = null;

// ==========================================
// HUMAN-LIKE UTILITIES
// ==========================================

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Add random variation to delays (±30%)
function humanWait(baseMs) {
  const variation = baseMs * 0.3;
  const randomMs = baseMs + (Math.random() * variation * 2 - variation);
  return wait(Math.floor(randomMs));
}

function waitForElement(selector, timeout = 5000) {
  return new Promise((resolve, reject) => {
    const element = document.querySelector(selector);
    if (element) return resolve(element);

    const observer = new MutationObserver(() => {
      const el = document.querySelector(selector);
      if (el) {
        observer.disconnect();
        resolve(el);
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });
    setTimeout(() => {
      observer.disconnect();
      reject(new Error(`Timeout: ${selector}`));
    }, timeout);
  });
}

// Human-like typing with random delays
async function humanTypeText(element, text) {
  element.focus();
  element.value = "";
  
  for (let char of text) {
    element.value += char;
    element.dispatchEvent(new Event("input", { bubbles: true }));
    await humanWait(TICKET_CONFIG.TYPING_SPEED);
  }
  
  element.dispatchEvent(new Event("change", { bubbles: true }));
  element.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true }));
}

function triggerInputEvent(element, value) {
  element.focus();
  element.value = value;
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
  element.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true }));
}

function isOnPassengerPage() {
  return document.querySelector('input[placeholder="Name"]') !== null;
}

// ==========================================
// ERROR DIALOG HANDLER
// ==========================================

function closeErrorDialogs() {
  const closeButtons = document.querySelectorAll(
    'button.ui-dialog-titlebar-icon, button.ui-dialog-titlebar-close, .ui-dialog .close, button[aria-label="Close"]'
  );

  closeButtons.forEach((btn) => {
    try {
      const dialog = btn.closest('.ui-dialog, .p-dialog, [role="dialog"]');
      if (dialog) {
        const errorText = dialog.textContent || "";
        if (errorText.includes("Error") || errorText.includes("request is under process")) {
          btn.click();
        }
      }
    } catch (e) {}
  });

  const overlays = document.querySelectorAll(".ui-widget-overlay, .p-dialog-mask");
  overlays.forEach((overlay) => {
    try {
      overlay.click();
    } catch (e) {}
  });
}

function startErrorDialogMonitor() {
  if (errorDialogInterval) return;
  errorDialogInterval = setInterval(() => {
    if (shouldStopRetrying) {
      stopErrorDialogMonitor();
      return;
    }
    closeErrorDialogs();
  }, RETRY_CONFIG.ERROR_DIALOG_CHECK_INTERVAL);
}

function stopErrorDialogMonitor() {
  if (errorDialogInterval) {
    clearInterval(errorDialogInterval);
    errorDialogInterval = null;
  }
}

// ==========================================
// RETRY UTILITIES
// ==========================================

async function retryUntilSuccess(actionFn, checkFn, interval, maxDuration, actionName) {
  const startTime = Date.now();

  while (!shouldStopRetrying && Date.now() - startTime < maxDuration) {
    try {
      if (checkFn()) {
        console.log(`✓ ${actionName} successful`);
        return true;
      }
      await actionFn();
      await wait(interval);
    } catch (err) {
      await wait(interval);
    }
  }

  if (shouldStopRetrying) return false;
  console.log(`Timeout: ${actionName}`);
  return false;
}

// ==========================================
// FORM FILLING FUNCTIONS
// ==========================================

async function fillStation(inputSelector, stationCode, stationFullMatch) {
  const input = await waitForElement(inputSelector, 2000);
  await humanWait(TICKET_CONFIG.CLICK_DELAY);
  
  await humanTypeText(input, stationCode);
  await humanWait(TICKET_CONFIG.ACTION_DELAY);

  try {
    await waitForElement(".ui-autocomplete-items li", 2000);
  } catch {
    triggerInputEvent(input, stationCode + " ");
    await waitForElement(".ui-autocomplete-items li", 2000);
  }

  await humanWait(TICKET_CONFIG.CLICK_DELAY);

  const listItems = document.querySelectorAll(".ui-autocomplete-items li");
  for (let item of listItems) {
    const itemText = item.innerText.trim();
    if (itemText.includes(stationFullMatch) || itemText.includes(stationCode)) {
      item.click();
      return;
    }
  }

  if (listItems.length > 0) listItems[0].click();
}

async function fillDateField(date) {
  const [day, month, year] = date.split("/").map((num) => parseInt(num));

  const dateInput = await waitForElement(".ui-calendar input", 2000);
  await humanWait(TICKET_CONFIG.CLICK_DELAY);
  dateInput.click();
  await humanWait(TICKET_CONFIG.ACTION_DELAY);

  await waitForElement(".ui-datepicker", 2000);

  const currentMonth = document.querySelector(".ui-datepicker-month");
  const currentYear = document.querySelector(".ui-datepicker-year");

  const monthNames = ["January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"];
  const targetMonthName = monthNames[month - 1];
  const targetYear = year.toString();

  let attempts = 0;
  while (attempts < 24) {
    const currMonth = currentMonth.textContent.trim();
    const currYear = currentYear.textContent.trim();

    if (currMonth === targetMonthName && currYear === targetYear) break;

    const currDate = new Date(parseInt(currYear), monthNames.indexOf(currMonth), 1);
    const targetDate = new Date(year, month - 1, 1);

    if (targetDate > currDate) {
      const nextBtn = document.querySelector(".ui-datepicker-next");
      if (nextBtn && !nextBtn.classList.contains("ui-state-disabled")) {
        nextBtn.click();
        await humanWait(200);
      }
    } else {
      const prevBtn = document.querySelector(".ui-datepicker-prev");
      if (prevBtn && !prevBtn.classList.contains("ui-state-disabled")) {
        prevBtn.click();
        await humanWait(200);
      }
    }

    attempts++;
  }

  const dayLinks = document.querySelectorAll(".ui-datepicker-calendar a.ui-state-default");
  for (let dayLink of dayLinks) {
    if (parseInt(dayLink.textContent.trim()) === day) {
      await humanWait(TICKET_CONFIG.CLICK_DELAY);
      dayLink.click();
      await humanWait(300);
      return;
    }
  }
}

async function selectDropdown(dropdownSelector, optionText) {
  try {
    const dropdown = await waitForElement(dropdownSelector, 200);
    await humanWait(TICKET_CONFIG.CLICK_DELAY);
    dropdown.click();
    await humanWait(TICKET_CONFIG.ACTION_DELAY);

    const options = document.querySelectorAll("p-dropdownitem li");
    for (let option of options) {
      if (option.innerText.trim() === optionText) {
        option.click();
        return;
      }
    }
  } catch {}
}

// ==========================================
// SEARCH & BOOKING
// ==========================================

async function clickSearchWithRetry() {
  const searchAction = async () => {
    const searchBtn = document.querySelector("button.search_btn");
    if (searchBtn && !searchBtn.disabled) {
      searchBtn.click();
    } else {
      throw new Error("Search button not available");
    }
  };

  const checkSuccess = () => document.querySelector("app-train-avl-enq") !== null;

  return await retryUntilSuccess(
    searchAction,
    checkSuccess,
    RETRY_CONFIG.SEARCH_BUTTON_INTERVAL,
    RETRY_CONFIG.MAX_RETRY_DURATION,
    "Search"
  );
}

async function findTrainByNumber(trainNumber) {
  await humanWait(1500);
  const trainElements = document.querySelectorAll("app-train-avl-enq");

  for (let trainEl of trainElements) {
    const trainHeading = trainEl.querySelector(".train-heading strong");
    if (trainHeading && trainHeading.textContent.includes(trainNumber)) {
      return trainEl;
    }
  }
  throw new Error(`Train ${trainNumber} not found`);
}

const CLASS_MAP = {
  SL: "Sleeper (SL)",
  "3E": "AC 3 Economy (3E)",
  "3A": "AC 3 Tier (3A)",
  "2A": "AC 2 Tier (2A)",
  "1A": "AC First Class (1A)",
};

async function selectTrainClass(trainElement, preferredClass, alternativeClasses = []) {
  const classesToTry = [preferredClass, ...alternativeClasses];

  for (let classCode of classesToTry) {
    const className = CLASS_MAP[classCode];
    if (!className) continue;

    const classSpans = trainElement.querySelectorAll("span.hidden-xs");
    for (let span of classSpans) {
      if (span.textContent.trim() === className) {
        await humanWait(TICKET_CONFIG.CLICK_DELAY);
        span.click();
        await humanWait(300);
        return { success: true, className, classCode };
      }
    }

    const classButtons = trainElement.querySelectorAll(".pre-avl");
    for (let button of classButtons) {
      const buttonText = button.querySelector("strong");
      if (buttonText && buttonText.textContent.includes(className)) {
        await humanWait(TICKET_CONFIG.CLICK_DELAY);
        button.click();
        await humanWait(300);
        return { success: true, className, classCode };
      }
    }
  }
  throw new Error("No available class found");
}

async function handleDatePopup(targetDate) {
  await humanWait(300);
  const dateOptions = document.querySelectorAll(".pre-avl");

  const [day, month] = targetDate.split("/");
  const targetDay = parseInt(day);
  const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const targetMonthName = monthNames[parseInt(month) - 1];

  const validDateOptions = [];
  for (let option of dateOptions) {
    const strongTag = option.querySelector("strong");
    if (!strongTag) continue;

    const dateText = strongTag.textContent.trim();
    if (dateText.match(/\w+,\s+\d+\s+\w+/)) {
      validDateOptions.push({ element: option, dateText });
    }
  }

  let selectedOption = null;
  for (let option of validDateOptions) {
    if (option.dateText.includes(`${targetDay} ${targetMonthName}`)) {
      selectedOption = option;
      break;
    }
  }

  if (!selectedOption && validDateOptions.length > 0) {
    selectedOption = validDateOptions[0];
  }

  if (selectedOption) {
    await humanWait(TICKET_CONFIG.CLICK_DELAY);
    selectedOption.element.click();
    await humanWait(1000);
    return { date: selectedOption.dateText };
  }
  throw new Error("No date option found");
}

async function clickBookNowWithRetry(trainElement) {
  const bookNowAction = async () => {
    let bookButton = trainElement.querySelector("button.btnDefault.train_Search");

    if (!bookButton) {
      const trainParent = trainElement.closest(".form-group");
      if (trainParent) {
        bookButton = trainParent.querySelector("button.btnDefault.train_Search");
      }
    }

    if (!bookButton) {
      const allButtons = trainElement.querySelectorAll("button");
      for (let button of allButtons) {
        if (button.textContent.includes("Book Now") && button.classList.contains("train_Search")) {
          bookButton = button;
          break;
        }
      }
    }

    if (!bookButton) throw new Error("Book Now button not found");

    bookButton.scrollIntoView({ behavior: "smooth", block: "center" });
    await humanWait(TICKET_CONFIG.CLICK_DELAY);
    bookButton.click();
  };

  const checkSuccess = () => isOnPassengerPage();

  return await retryUntilSuccess(
    bookNowAction,
    checkSuccess,
    RETRY_CONFIG.BOOK_NOW_INTERVAL,
    RETRY_CONFIG.MAX_RETRY_DURATION,
    "Book Now"
  );
}

// ==========================================
// PASSENGER FORM
// ==========================================

async function fillPassengerDetails(passengers) {
  await humanWait(1500);

  for (let i = 0; i < passengers.length; i++) {
    const p = passengers[i];

    const nameInputs = document.querySelectorAll('input[placeholder="Name"]');
    if (nameInputs[i]) {
      await humanTypeText(nameInputs[i], p.name);
      await humanWait(200);
      nameInputs[i].dispatchEvent(new KeyboardEvent("keydown", {
        key: "Escape", keyCode: 27, bubbles: true
      }));
      await humanWait(100);
    }

    const ageInputs = document.querySelectorAll('input[placeholder="Age"]');
    if (ageInputs[i]) {
      triggerInputEvent(ageInputs[i], p.age.toString());
      await humanWait(100);
    }

    const genderSelects = document.querySelectorAll('select[formcontrolname="passengerGender"]');
    if (genderSelects[i]) {
      await humanWait(TICKET_CONFIG.CLICK_DELAY);
      genderSelects[i].value = p.gender;
      genderSelects[i].dispatchEvent(new Event("change", { bubbles: true }));
      await humanWait(100);
    }

    if (p.berthPreference) {
      const berthSelects = document.querySelectorAll('select[formcontrolname="passengerBerthChoice"]');
      if (berthSelects[i]) {
        await humanWait(TICKET_CONFIG.CLICK_DELAY);
        berthSelects[i].value = p.berthPreference;
        berthSelects[i].dispatchEvent(new Event("change", { bubbles: true }));
        await humanWait(100);
      }
    }

    if (p.nationality) {
      try {
        const nationalitySelects = document.querySelectorAll('select[formcontrolname="passengerNationality"]');
        if (nationalitySelects[i]) {
          nationalitySelects[i].value = p.nationality;
          nationalitySelects[i].dispatchEvent(new Event("change", { bubbles: true }));
        }
      } catch {}
    }
  }
}

async function setBookingPreferences(autoUpgrade, confirmBerthsOnly) {
  try {
    if (autoUpgrade) {
      const autoUpgradeCheckbox = document.querySelector('input[id="autoUpgradation"]');
      if (autoUpgradeCheckbox && !autoUpgradeCheckbox.checked) {
        await humanWait(TICKET_CONFIG.CLICK_DELAY);
        autoUpgradeCheckbox.click();
      }
    }

    if (confirmBerthsOnly) {
      const confirmBerthsCheckbox = document.querySelector('input[id="confirmberths"]');
      if (confirmBerthsCheckbox && !confirmBerthsCheckbox.checked) {
        await humanWait(TICKET_CONFIG.CLICK_DELAY);
        confirmBerthsCheckbox.click();
      }
    }
  } catch {}
}

async function selectPaymentMethod(paymentType, upiId = null) {
  try {
    const loyaltyRadio = document.querySelector('input[name="loyalityOperationType"][value="2"]');
    if (loyaltyRadio && loyaltyRadio.checked) {
      const loyaltyContainer = loyaltyRadio.closest("p-radiobutton");
      if (loyaltyContainer) {
        await humanWait(TICKET_CONFIG.CLICK_DELAY);
        loyaltyContainer.click();
      }
    }

    await humanWait(TICKET_CONFIG.ACTION_DELAY);

    if (paymentType === "UPI") {
      const upiRadio = document.querySelector('input[name="paymentType"][value="2"]');
      if (upiRadio && !upiRadio.checked) {
        await humanWait(TICKET_CONFIG.CLICK_DELAY);
        upiRadio.click();
        await humanWait(100);

        const upiContainer = upiRadio.closest("p-radiobutton");
        if (upiContainer && !upiRadio.checked) {
          const radioBox = upiContainer.querySelector(".ui-radiobutton-box");
          if (radioBox) radioBox.click();
        }

        await humanWait(TICKET_CONFIG.ACTION_DELAY);

        if (upiId) {
          const upiInputSelectors = [
            'input[placeholder*="UPI"]',
            'input[formcontrolname="vpaId"]',
            'input[name*="upi"]',
          ];

          for (let selector of upiInputSelectors) {
            const upiInput = document.querySelector(selector);
            if (upiInput) {
              await humanTypeText(upiInput, upiId);
              break;
            }
          }
        }
      }
    }
  } catch {}
}

async function clickContinueToPayment() {
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

  if (!continueButton) throw new Error("Continue button not found");

  continueButton.scrollIntoView({ behavior: "smooth", block: "center" });
  await humanWait(TICKET_CONFIG.CLICK_DELAY);
  continueButton.click();

  shouldStopRetrying = true;
  stopErrorDialogMonitor();
  await wait(500);
  console.log("🛑 Script completed successfully");
}

// ==========================================
// MAIN EXECUTION
// ==========================================

async function fillIRCTCForm() {
  shouldStopRetrying = false;

  try {
    console.log("=== Starting IRCTC Booking ===");
    startErrorDialogMonitor();

    await fillStation(
      'input[aria-label="Enter From station. Input is Mandatory."]',
      TICKET_CONFIG.FROM_STATION_CODE,
      TICKET_CONFIG.FROM_STATION_FULL
    );
    await humanWait(TICKET_CONFIG.ACTION_DELAY);

    await fillStation(
      'input[aria-label="Enter To station. Input is Mandatory."]',
      TICKET_CONFIG.TO_STATION_CODE,
      TICKET_CONFIG.TO_STATION_FULL
    );
    await humanWait(TICKET_CONFIG.ACTION_DELAY);

    await fillDateField(TICKET_CONFIG.TRAVEL_DATE);
    await humanWait(TICKET_CONFIG.ACTION_DELAY);

    if (TICKET_CONFIG.TRAIN_CLASS && TICKET_CONFIG.TRAIN_CLASS !== "All Classes") {
      await selectDropdown(".ng-tns-c76-10.ui-dropdown", TICKET_CONFIG.TRAIN_CLASS);
    }

    if (TICKET_CONFIG.QUOTA && TICKET_CONFIG.QUOTA !== "GENERAL") {
      await selectDropdown(".ng-tns-c76-11.ui-dropdown", TICKET_CONFIG.QUOTA);
    }
    await humanWait(TICKET_CONFIG.ACTION_DELAY);

    const searchSuccess = await clickSearchWithRetry();
    if (!searchSuccess) throw new Error("Search failed");

    const trainElement = await findTrainByNumber(TICKET_CONFIG.TARGET_TRAIN_NUMBER);
    await selectTrainClass(trainElement, TICKET_CONFIG.PREFERRED_CLASS, TICKET_CONFIG.ALTERNATIVE_CLASSES);
    await handleDatePopup(TICKET_CONFIG.TRAVEL_DATE);

    const bookSuccess = await clickBookNowWithRetry(trainElement);
    if (!bookSuccess) throw new Error("Booking failed");

    await fillPassengerDetails(TICKET_CONFIG.PASSENGERS);
    await humanWait(TICKET_CONFIG.ACTION_DELAY);
    await setBookingPreferences(TICKET_CONFIG.AUTO_UPGRADATION, TICKET_CONFIG.CONFIRM_BERTHS_ONLY);
    await humanWait(TICKET_CONFIG.ACTION_DELAY);
    await selectPaymentMethod(TICKET_CONFIG.PAYMENT_TYPE, TICKET_CONFIG.UPI_ID);
    await humanWait(800);

    await clickContinueToPayment();
  } catch (err) {
    console.error("❌ Error:", err.message);
    shouldStopRetrying = true;
    stopErrorDialogMonitor();
  }
}

window.stopIRCTCAutofill = () => {
  shouldStopRetrying = true;
  stopErrorDialogMonitor();
  console.log("⚠ Stopped manually");
};

fillIRCTCForm();
window.irctcAutofill = fillIRCTCForm;
// ==========================================
// IRCTC FORM AUTOFILL - OPTIMIZED FOR SPEED
// ==========================================

const TICKET_CONFIG = {
  FROM_STATION_CODE: "GZB",
  FROM_STATION_FULL: "GHAZIABAD - GZB",
  TO_STATION_CODE: "HRI",
  TO_STATION_FULL: "HARDOI - HRI",
  TRAVEL_DATE: "20/12/2025", // Format: DD/MM/YYYY
  TRAIN_CLASS: "All Classes", // Options: 'All Classes', 'Sleeper (SL)', '3A (3A)', etc.
  QUOTA: "TATKAL", // Options: 'GENERAL', 'LADIES', 'TATKAL', etc.
};

// Speed optimizations: Reduced all wait times to minimum needed for DOM updates

// ==========================================
// HELPER UTILITIES
// ==========================================

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function waitForElement(selector, timeout = 2000) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(selector)) {
      return resolve(document.querySelector(selector));
    }

    const observer = new MutationObserver(() => {
      if (document.querySelector(selector)) {
        observer.disconnect();
        resolve(document.querySelector(selector));
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });

    setTimeout(() => {
      observer.disconnect();
      reject(new Error(`Timeout waiting for ${selector}`));
    }, timeout);
  });
}

// Angular inputs require event dispatching to recognize changes
function triggerInputEvent(element, value) {
  element.focus();
  element.value = value;
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
  element.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true }));
  element.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true }));
}

// ==========================================
// STATION FILLING (FROM FIRST CODE)
// ==========================================

async function fillStation(inputSelector, stationCode, stationFullMatch) {
  console.log(`📍 Filling station: ${stationCode}...`);

  try {
    const input = await waitForElement(inputSelector, 2000);

    if (!input) {
      throw new Error("Station input not found!");
    }

    // Type the Station Code
    triggerInputEvent(input, stationCode);

    // Wait for the autocomplete dropdown (reduced wait)
    await wait(200);

    try {
      await waitForElement(".ui-autocomplete-items li", 2000);
    } catch (e) {
      console.warn("Dropdown didn't appear, retrying...");
      triggerInputEvent(input, stationCode + " ");
      await waitForElement(".ui-autocomplete-items li", 2000);
    }

    // Find the correct list item and click it
    const listItems = document.querySelectorAll(".ui-autocomplete-items li");
    let found = false;

    for (let item of listItems) {
      const itemText = item.innerText.trim();
      if (
        itemText.includes(stationFullMatch) ||
        itemText.includes(stationCode)
      ) {
        console.log(`✅ Found match: ${itemText}`);
        item.click();
        found = true;
        break;
      }
    }

    if (!found && listItems.length > 0) {
      console.log("⚠️ Selecting first suggestion");
      listItems[0].click();
    }

    await wait(100);
    console.log(`✓ Station filled: ${stationCode}`);
  } catch (error) {
    console.error(`❌ Error filling station ${stationCode}:`, error.message);
    throw error;
  }
}

// ==========================================
// DATE FILLING (FROM SECOND CODE)
// ==========================================

async function fillDateField(date) {
  console.log(`📅 Filling date: ${date}...`);

  try {
    // Parse date (DD/MM/YYYY)
    const [day, month, year] = date.split("/").map((num) => parseInt(num));

    // Click on date input to open calendar
    const dateInput = await waitForElement(".ui-calendar input", 2000);
    dateInput.click();
    await wait(300);

    // Wait for calendar to appear
    await waitForElement(".ui-datepicker", 2000);

    // Get current month and year elements
    const currentMonth = document.querySelector(".ui-datepicker-month");
    const currentYear = document.querySelector(".ui-datepicker-year");

    if (!currentMonth || !currentYear) {
      throw new Error("Calendar month/year selectors not found");
    }

    const monthNames = [
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December",
    ];

    const targetMonthName = monthNames[month - 1];
    const targetYear = year.toString();

    // Navigate to correct month/year
    let attempts = 0;
    while (attempts < 24) {
      const currMonth = currentMonth.textContent.trim();
      const currYear = currentYear.textContent.trim();

      if (currMonth === targetMonthName && currYear === targetYear) {
        console.log(`✓ Navigated to: ${currMonth} ${currYear}`);
        break;
      }

      // Determine direction
      const currDate = new Date(
        parseInt(currYear),
        monthNames.indexOf(currMonth),
        1
      );
      const targetDate = new Date(year, month - 1, 1);

      if (targetDate > currDate) {
        const nextBtn = document.querySelector(".ui-datepicker-next");
        if (nextBtn && !nextBtn.classList.contains("ui-state-disabled")) {
          nextBtn.click();
        }
      } else {
        const prevBtn = document.querySelector(".ui-datepicker-prev");
        if (prevBtn && !prevBtn.classList.contains("ui-state-disabled")) {
          prevBtn.click();
        }
      }

      await wait(100);
      attempts++;
    }

    if (attempts >= 24) {
      throw new Error("Could not navigate to target month");
    }

    // Click on the specific day
    const dayLinks = document.querySelectorAll(
      ".ui-datepicker-calendar a.ui-state-default"
    );
    let dayFound = false;

    for (let dayLink of dayLinks) {
      const dayText = dayLink.textContent.trim();
      if (parseInt(dayText) === day) {
        console.log(`✓ Clicking day: ${day}`);
        dayLink.click();
        dayFound = true;
        await wait(150);
        break;
      }
    }

    if (!dayFound) {
      throw new Error(`Day ${day} not found in calendar`);
    }

    console.log(`✓ Date filled: ${date}`);
  } catch (error) {
    console.error("❌ Error filling date:", error.message);
    throw error;
  }
}

// ==========================================
// DROPDOWN SELECTION
// ==========================================

async function selectDropdown(dropdownSelector, optionText) {
  console.log(`🔽 Selecting dropdown option: ${optionText}...`);

  try {
    const dropdown = await waitForElement(dropdownSelector, 2000);
    dropdown.click();
    await wait(200);

    const options = document.querySelectorAll("p-dropdownitem li");
    let found = false;

    for (let option of options) {
      if (option.innerText.trim() === optionText) {
        option.click();
        found = true;
        console.log(`✓ Selected: ${optionText}`);
        break;
      }
    }

    if (!found) {
      console.warn(`⚠️ Option "${optionText}" not found, skipping`);
    }

    await wait(100);
  } catch (error) {
    console.warn(`⚠️ Could not select dropdown: ${error.message}`);
  }
}

// ==========================================
// MAIN EXECUTION
// ==========================================

async function fillIRCTCForm() {
  console.log("🚀 Starting IRCTC Auto-Fill...");
  console.log("═".repeat(50));

  try {
    // STEP 1: Fill FROM Station
    await fillStation(
      'input[aria-label="Enter From station. Input is Mandatory."]',
      TICKET_CONFIG.FROM_STATION_CODE,
      TICKET_CONFIG.FROM_STATION_FULL
    );

    await wait(150);

    // STEP 2: Fill TO Station
    await fillStation(
      'input[aria-label="Enter To station. Input is Mandatory."]',
      TICKET_CONFIG.TO_STATION_CODE,
      TICKET_CONFIG.TO_STATION_FULL
    );

    await wait(150);

    // STEP 3: Fill Journey Date
    await fillDateField(TICKET_CONFIG.TRAVEL_DATE);

    await wait(150);

    // STEP 4: Select Class (if not "All Classes")
    if (
      TICKET_CONFIG.TRAIN_CLASS &&
      TICKET_CONFIG.TRAIN_CLASS !== "All Classes"
    ) {
      await selectDropdown(
        ".ng-tns-c76-10.ui-dropdown",
        TICKET_CONFIG.TRAIN_CLASS
      );
    }

    // STEP 5: Select Quota (if not "GENERAL")
    if (TICKET_CONFIG.QUOTA && TICKET_CONFIG.QUOTA !== "GENERAL") {
      await selectDropdown(".ng-tns-c76-11.ui-dropdown", TICKET_CONFIG.QUOTA);
    }

    await wait(200);

    // STEP 6: Click Search Button
    const searchBtn = await waitForElement("button.search_btn", 2000);
    if (searchBtn) {
      console.log("🔍 Clicking Search button...");
      searchBtn.click();
      console.log("✓ Search button clicked!");
    } else {
      throw new Error("Search button not found");
    }

    console.log("═".repeat(50));
    console.log("✅ AUTOFILL COMPLETED SUCCESSFULLY!");
    console.log("═".repeat(50));
  } catch (err) {
    console.error("═".repeat(50));
    console.error("❌ AUTOFILL FAILED:", err.message);
    console.error("═".repeat(50));
  }
}

// Run the script automatically
fillIRCTCForm();

// Also expose it globally for manual execution
window.irctcAutofill = fillIRCTCForm;

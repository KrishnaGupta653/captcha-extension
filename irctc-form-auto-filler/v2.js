// ==========================================
// IRCTC FORM AUTOFILL + TRAIN BOOKING - COMPLETE FLOW
// ==========================================

const TICKET_CONFIG = {
  FROM_STATION_CODE: "GZB",
  FROM_STATION_FULL: "GHAZIABAD - GZB",
  TO_STATION_CODE: "HRI",
  TO_STATION_FULL: "HARDOI - HRI",
  TRAVEL_DATE: "20/12/2025", // Format: DD/MM/YYYY
  TRAIN_CLASS: "All Classes", // Options: 'All Classes', 'Sleeper (SL)', '3A (3A)', etc.
  QUOTA: "TATKAL", // Options: 'GENERAL', 'LADIES', 'TATKAL', etc.
  
  // NEW: Train Selection Config
  TARGET_TRAIN_NUMBER: "12230", // Train number to search for and book
  PREFERRED_CLASS: "3A", // Preferred class: "SL", "3E", "3A", "2A", "1A"
  // Alternative classes in order of preference if primary not available
  ALTERNATIVE_CLASSES: ["SL", "3E", "2A", "1A"]
};

// ==========================================
// HELPER UTILITIES
// ==========================================

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function log(message, type = "info") {
  const emoji = {
    info: "ℹ️",
    success: "✅",
    error: "❌",
    warning: "⚠️",
    search: "🔍",
    click: "👆",
    wait: "⏳"
  };
  console.log(`${emoji[type] || "📍"} ${message}`);
}

function waitForElement(selector, timeout = 5000) {
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

function waitForElementDisappear(selector, timeout = 5000) {
  return new Promise((resolve, reject) => {
    if (!document.querySelector(selector)) {
      return resolve();
    }

    const observer = new MutationObserver(() => {
      if (!document.querySelector(selector)) {
        observer.disconnect();
        resolve();
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });

    setTimeout(() => {
      observer.disconnect();
      reject(new Error(`Timeout waiting for ${selector} to disappear`));
    }, timeout);
  });
}

function triggerInputEvent(element, value) {
  element.focus();
  element.value = value;
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
  element.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true }));
  element.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true }));
}

// ==========================================
// STATION FILLING
// ==========================================

async function fillStation(inputSelector, stationCode, stationFullMatch) {
  log(`Filling station: ${stationCode}...`, "info");

  try {
    const input = await waitForElement(inputSelector, 2000);

    if (!input) {
      throw new Error("Station input not found!");
    }

    triggerInputEvent(input, stationCode);
    await wait(200);

    try {
      await waitForElement(".ui-autocomplete-items li", 2000);
    } catch (e) {
      log("Dropdown didn't appear, retrying...", "warning");
      triggerInputEvent(input, stationCode + " ");
      await waitForElement(".ui-autocomplete-items li", 2000);
    }

    const listItems = document.querySelectorAll(".ui-autocomplete-items li");
    let found = false;

    for (let item of listItems) {
      const itemText = item.innerText.trim();
      if (
        itemText.includes(stationFullMatch) ||
        itemText.includes(stationCode)
      ) {
        log(`Found match: ${itemText}`, "success");
        item.click();
        found = true;
        break;
      }
    }

    if (!found && listItems.length > 0) {
      log("Selecting first suggestion", "warning");
      listItems[0].click();
    }

    await wait(100);
    log(`Station filled: ${stationCode}`, "success");
  } catch (error) {
    log(`Error filling station ${stationCode}: ${error.message}`, "error");
    throw error;
  }
}

// ==========================================
// DATE FILLING
// ==========================================

async function fillDateField(date) {
  log(`Filling date: ${date}...`, "info");

  try {
    const [day, month, year] = date.split("/").map((num) => parseInt(num));

    const dateInput = await waitForElement(".ui-calendar input", 2000);
    dateInput.click();
    await wait(300);

    await waitForElement(".ui-datepicker", 2000);

    const currentMonth = document.querySelector(".ui-datepicker-month");
    const currentYear = document.querySelector(".ui-datepicker-year");

    if (!currentMonth || !currentYear) {
      throw new Error("Calendar month/year selectors not found");
    }

    const monthNames = [
      "January", "February", "March", "April", "May", "June",
      "July", "August", "September", "October", "November", "December",
    ];

    const targetMonthName = monthNames[month - 1];
    const targetYear = year.toString();

    let attempts = 0;
    while (attempts < 24) {
      const currMonth = currentMonth.textContent.trim();
      const currYear = currentYear.textContent.trim();

      if (currMonth === targetMonthName && currYear === targetYear) {
        log(`Navigated to: ${currMonth} ${currYear}`, "success");
        break;
      }

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

    const dayLinks = document.querySelectorAll(
      ".ui-datepicker-calendar a.ui-state-default"
    );
    let dayFound = false;

    for (let dayLink of dayLinks) {
      const dayText = dayLink.textContent.trim();
      if (parseInt(dayText) === day) {
        log(`Clicking day: ${day}`, "click");
        dayLink.click();
        dayFound = true;
        await wait(150);
        break;
      }
    }

    if (!dayFound) {
      throw new Error(`Day ${day} not found in calendar`);
    }

    log(`Date filled: ${date}`, "success");
  } catch (error) {
    log(`Error filling date: ${error.message}`, "error");
    throw error;
  }
}

// ==========================================
// DROPDOWN SELECTION
// ==========================================

async function selectDropdown(dropdownSelector, optionText) {
  log(`Selecting dropdown option: ${optionText}...`, "info");

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
        log(`Selected: ${optionText}`, "success");
        break;
      }
    }

    if (!found) {
      log(`Option "${optionText}" not found, skipping`, "warning");
    }

    await wait(100);
  } catch (error) {
    log(`Could not select dropdown: ${error.message}`, "warning");
  }
}

// ==========================================
// TRAIN SEARCH AND BOOKING FUNCTIONS
// ==========================================

async function findTrainByNumber(trainNumber) {
  log(`Searching for train: ${trainNumber}...`, "search");
  
  try {
    // Wait for train list to load
    await wait(1000);
    
    // Find all train elements
    const trainElements = document.querySelectorAll("app-train-avl-enq");
    
    for (let trainEl of trainElements) {
      const trainHeading = trainEl.querySelector(".train-heading strong");
      if (trainHeading && trainHeading.textContent.includes(trainNumber)) {
        log(`Found train: ${trainHeading.textContent}`, "success");
        return trainEl;
      }
    }
    
    throw new Error(`Train ${trainNumber} not found in the list`);
  } catch (error) {
    log(`Error finding train: ${error.message}`, "error");
    throw error;
  }
}

async function getClassMappings() {
  return {
    "SL": "Sleeper (SL)",
    "3E": "AC 3 Economy (3E)",
    "3A": "AC 3 Tier (3A)",
    "2A": "AC 2 Tier (2A)",
    "1A": "AC First Class (1A)"
  };
}

async function selectTrainClass(trainElement, preferredClass, alternativeClasses = []) {
  log(`Selecting class: ${preferredClass}...`, "info");
  
  try {
    const classMappings = await getClassMappings();
    const classesToTry = [preferredClass, ...alternativeClasses];
    
    for (let classCode of classesToTry) {
      const className = classMappings[classCode];
      if (!className) continue;
      
      log(`Trying class: ${className}`, "search");
      
      // Find the class span with "hidden-xs" class within the train element
      const classSpans = trainElement.querySelectorAll("span.hidden-xs");
      
      for (let span of classSpans) {
        const spanText = span.textContent.trim();
        if (spanText === className) {
          log(`Found class span: ${className}`, "success");
          
          // Click the span to open date popup
          log(`Clicking "${className}" to open date popup...`, "click");
          span.click();
          await wait(1500); // Wait for popup to appear
          
          return { success: true, className, classCode };
        }
      }
      
      // Fallback: Try clicking the .pre-avl div if span not found
      const classButtons = trainElement.querySelectorAll(".pre-avl");
      
      for (let button of classButtons) {
        const buttonText = button.querySelector("strong");
        if (buttonText && buttonText.textContent.includes(className)) {
          log(`Found class button (fallback): ${className}`, "success");
          
          log(`Clicking ${className} button to open date popup...`, "click");
          button.click();
          await wait(1500);
          
          return { success: true, className, classCode };
        }
      }
    }
    
    throw new Error("No available class found");
  } catch (error) {
    log(`Error selecting class: ${error.message}`, "error");
    throw error;
  }
}

async function handleDatePopup(targetDate) {
  log("Waiting for date selection popup...", "wait");
  
  try {
    // Wait for popup to fully render
    await wait(1500);
    
    // Look for the date popup - it appears as an overlay or within the train element
    // Find all .pre-avl elements that contain dates
    const dateOptions = document.querySelectorAll(".pre-avl");
    
    if (dateOptions.length === 0) {
      log("No date options found", "warning");
      return null;
    }
    
    log(`Found ${dateOptions.length} date option elements, searching for dates...`, "search");
    
    // Parse target date
    const [day, month, year] = targetDate.split("/");
    const targetDay = parseInt(day);
    const targetMonthName = getMonthName(parseInt(month));
    
    // Filter to find date options (ones with date text like "Sat, 20 Dec")
    const validDateOptions = [];
    
    for (let option of dateOptions) {
      const strongTag = option.querySelector("strong");
      if (!strongTag) continue;
      
      const dateText = strongTag.textContent.trim();
      
      // Check if this looks like a date (contains day name and number)
      if (dateText.match(/\w+,\s+\d+\s+\w+/)) {
        
        // Get the status (WL12, REGRET, AVAILABLE, etc.)
        const statusDivs = option.querySelectorAll("div[class*='col-xs-12']");
        let status = "Unknown";
        
        for (let statusDiv of statusDivs) {
          const statusText = statusDiv.textContent.trim();
          if (statusText && statusText !== dateText) {
            status = statusText;
            break;
          }
        }
        
        validDateOptions.push({
          element: option,
          dateText: dateText,
          status: status
        });
        
        log(`Date option: ${dateText} - Status: ${status}`, "info");
      }
    }
    
    if (validDateOptions.length === 0) {
      log("No valid date options found in popup", "warning");
      return null;
    }
    
    // Try to find the date that matches our target (Sat, 20 Dec)
    let selectedOption = null;
    
    for (let option of validDateOptions) {
      // Check if date matches (looking for day number and month)
      if (option.dateText.includes(`${targetDay} ${targetMonthName}`)) {
        log(`Found matching date: ${option.dateText}`, "success");
        selectedOption = option;
        break;
      }
    }
    
    // If exact date not found, select first available option
    if (!selectedOption && validDateOptions.length > 0) {
      log("Exact date not found, selecting first available option", "warning");
      selectedOption = validDateOptions[0];
    }
    
    if (selectedOption) {
      log(`Selecting: ${selectedOption.dateText} (${selectedOption.status})`, "click");
      
      // Add selected-class to the element
      selectedOption.element.classList.add("selected-class");
      
      // Click the date option
      selectedOption.element.click();
      await wait(1000);
      
      log("✅ Date selected from popup!", "success");
      return { 
        date: selectedOption.dateText, 
        status: selectedOption.status 
      };
    } else {
      throw new Error("Could not find any date option to select");
    }
    
  } catch (error) {
    log(`Error handling date popup: ${error.message}`, "error");
    throw error;
  }
}

function getMonthName(monthNum) {
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return months[monthNum - 1];
}

async function clickBookNow(trainElement) {
  log("Looking for Book Now button...", "search");
  
  try {
    // Wait a moment for the date selection to register
    await wait(800);
    
    // Find Book Now button with specific classes: btnDefault train_Search
    // First try within the train element
    let bookButton = trainElement.querySelector("button.btnDefault.train_Search");
    
    if (!bookButton) {
      // Try finding it in the parent container
      log("Button not in train element, searching parent...", "search");
      const trainParent = trainElement.closest(".form-group");
      if (trainParent) {
        bookButton = trainParent.querySelector("button.btnDefault.train_Search");
      }
    }
    
    if (!bookButton) {
      // Fallback: search all Book Now buttons
      log("Trying fallback search...", "search");
      const allButtons = trainElement.querySelectorAll("button");
      
      for (let button of allButtons) {
        const buttonText = button.textContent.trim();
        if ((buttonText.includes("Book Now") || buttonText.includes("BOOK NOW")) 
            && button.classList.contains("train_Search")) {
          bookButton = button;
          break;
        }
      }
    }
    
    if (!bookButton) {
      throw new Error("Book Now button not found");
    }
    
    log("✅ Found Book Now button!", "success");
    
    // Check if button is disabled
    if (bookButton.classList.contains("disable-book") || bookButton.disabled) {
      log("⚠️  Book Now button is disabled", "warning");
      log("This usually means no seats available or REGRET status", "warning");
      log("Attempting to click anyway...", "info");
    }
    
    // Scroll button into view
    bookButton.scrollIntoView({ behavior: "smooth", block: "center" });
    await wait(500);
    
    log("🎯 Clicking Book Now button...", "click");
    bookButton.click();
    
    // Wait for page transition
    await wait(2500);
    
    // Check if we moved to passenger details page
    const currentUrl = window.location.href;
    if (currentUrl.includes("passenger") || currentUrl.includes("booking") || 
        currentUrl.includes("book")) {
      log("✅ Successfully navigated to booking/passenger page!", "success");
    } else {
      log("⏳ Waiting for page to load...", "wait");
      await wait(2000);
      
      // Check again
      const newUrl = window.location.href;
      if (newUrl !== currentUrl) {
        log(`✅ Page changed to: ${newUrl}`, "success");
      } else {
        log("⚠️  Page did not change - booking may have failed", "warning");
      }
    }
    
    return true;
    
  } catch (error) {
    log(`❌ Error clicking Book Now: ${error.message}`, "error");
    throw error;
  }
}

// ==========================================
// MAIN EXECUTION - COMPLETE FLOW
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
      log("Clicking Search button...", "click");
      searchBtn.click();
      log("Search button clicked!", "success");
    } else {
      throw new Error("Search button not found");
    }

    console.log("═".repeat(50));
    log("FORM FILLING COMPLETED! Now searching for train...", "success");
    console.log("═".repeat(50));

    // Wait for results page to load
    await wait(3000);

    // STEP 7: Find the specific train
    const trainElement = await findTrainByNumber(TICKET_CONFIG.TARGET_TRAIN_NUMBER);

    // STEP 8: Select the train class (this opens the date popup)
    const classSelection = await selectTrainClass(
      trainElement,
      TICKET_CONFIG.PREFERRED_CLASS,
      TICKET_CONFIG.ALTERNATIVE_CLASSES
    );

    log(`Selected class: ${classSelection.className}`, "success");

    // STEP 9: Handle date popup that appears after clicking class
    const dateSelection = await handleDatePopup(TICKET_CONFIG.TRAVEL_DATE);
    if (dateSelection) {
      log(`Selected: ${dateSelection.date} - Status: ${dateSelection.status}`, "success");
    }

    // STEP 10: Click Book Now
    await clickBookNow(trainElement);

    console.log("═".repeat(50));
    log("BOOKING FLOW COMPLETED SUCCESSFULLY!", "success");
    log("Proceeding to passenger details page...", "info");
    console.log("═".repeat(50));

  } catch (err) {
    console.log("═".repeat(50));
    log(`AUTOFILL FAILED: ${err.message}`, "error");
    console.log("═".repeat(50));
    console.error("Full error:", err);
  }
}

// ==========================================
// INITIALIZE
// ==========================================

// Run the script automatically
fillIRCTCForm();

// Also expose it globally for manual execution
window.irctcAutofill = fillIRCTCForm;

// Expose individual functions for debugging
window.irctcDebug = {
  findTrain: findTrainByNumber,
  selectClass: selectTrainClass,
  clickBook: clickBookNow,
  handlePopup: handleDatePopup
};

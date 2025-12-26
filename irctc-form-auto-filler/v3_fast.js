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
  QUOTA: "PREMIUM TATKAL", // Options: 'GENERAL', 'LADIES', 'TATKAL', etc.

  // NEW: Train Selection Config
  TARGET_TRAIN_NUMBER: "12230", // Train number to search for and book
  PREFERRED_CLASS: "3A", // Preferred class: "SL", "3E", "3A", "2A", "1A"
  // Alternative classes in order of preference if primary not available
  ALTERNATIVE_CLASSES: ["SL", "3E", "2A", "1A"],
  PASSENGERS: [
    {
      name: "Ayush Chowdhary",
      age: 30,
      gender: "M", // M/F/T
      berthPreference: "LB", // LB/MB/UB/SL/SU or "" for no preference
      foodChoice: "V", // V for Veg, N for Non-veg, "" for no choice
      nationality: "IN",
    },
    // Add more passengers if needed
    // {
    //   name: "JANE DOE",
    //   age: 28,
    //   gender: "F",
    //   berthPreference: "LB",
    //   foodChoice: "V",
    //   nationality: "IN"
    // }
  ],
  AUTO_UPGRADATION: true, // true/false
  CONFIRM_BERTHS_ONLY: true, // true/false

  // Payment Details
  PAYMENT_TYPE: "UPI", // "UPI" or other options
  USE_LOYALTY_POINTS: false,
  UPI_ID: "yourname@paytm",
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
    wait: "⏳",
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
    const dropdown = await waitForElement(dropdownSelector, 200);
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
    await wait(100);

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
    SL: "Sleeper (SL)",
    "3E": "AC 3 Economy (3E)",
    "3A": "AC 3 Tier (3A)",
    "2A": "AC 2 Tier (2A)",
    "1A": "AC First Class (1A)",
  };
}

async function selectTrainClass(
  trainElement,
  preferredClass,
  alternativeClasses = []
) {
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
          await wait(150); // Wait for popup to appear

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
          await wait(150);

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
    await wait(150);

    // Look for the date popup - it appears as an overlay or within the train element
    // Find all .pre-avl elements that contain dates
    const dateOptions = document.querySelectorAll(".pre-avl");

    if (dateOptions.length === 0) {
      log("No date options found", "warning");
      return null;
    }

    log(
      `Found ${dateOptions.length} date option elements, searching for dates...`,
      "search"
    );

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
          status: status,
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
      log(
        `Selecting: ${selectedOption.dateText} (${selectedOption.status})`,
        "click"
      );

      // Add selected-class to the element
      selectedOption.element.classList.add("selected-class");

      // Click the date option
      selectedOption.element.click();
      await wait(1000);

      log("✅ Date selected from popup!", "success");
      return {
        date: selectedOption.dateText,
        status: selectedOption.status,
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
  const months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];
  return months[monthNum - 1];
}

async function clickBookNow(trainElement) {
  log("Looking for Book Now button...", "search");

  try {
    // Wait a moment for the date selection to register
    await wait(200);

    // Find Book Now button with specific classes: btnDefault train_Search
    // First try within the train element
    let bookButton = trainElement.querySelector(
      "button.btnDefault.train_Search"
    );

    if (!bookButton) {
      // Try finding it in the parent container
      log("Button not in train element, searching parent...", "search");
      const trainParent = trainElement.closest(".form-group");
      if (trainParent) {
        bookButton = trainParent.querySelector(
          "button.btnDefault.train_Search"
        );
      }
    }

    if (!bookButton) {
      // Fallback: search all Book Now buttons
      log("Trying fallback search...", "search");
      const allButtons = trainElement.querySelectorAll("button");

      for (let button of allButtons) {
        const buttonText = button.textContent.trim();
        if (
          (buttonText.includes("Book Now") ||
            buttonText.includes("BOOK NOW")) &&
          button.classList.contains("train_Search")
        ) {
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
    await wait(200);

    log("🎯 Clicking Book Now button...", "click");
    bookButton.click();

    // Wait for page transition
    await wait(250);

    // Check if we moved to passenger details page
    const currentUrl = window.location.href;
    if (
      currentUrl.includes("passenger") ||
      currentUrl.includes("booking") ||
      currentUrl.includes("book")
    ) {
      log("✅ Successfully navigated to booking/passenger page!", "success");
    } else {
      log("⏳ Waiting for page to load...", "wait");
      await wait(200);

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
// PASSENGER FORM FILLING FUNCTIONS
// ==========================================

async function fillPassengerDetails(passengers) {
  log("Starting passenger form filling...", "info");

  try {
    // Wait for passenger form to load
    await wait(100);

    for (let i = 0; i < passengers.length; i++) {
      const passenger = passengers[i];
      log(`Filling passenger ${i + 1}: ${passenger.name}`, "info");

      // Fill name
      await fillPassengerName(i, passenger.name);
      await wait(50);

      // Fill age
      await fillPassengerAge(i, passenger.age);
      await wait(50);
      // Select gender
      await selectPassengerGender(i, passenger.gender);
      await wait(50);

      // Select berth preference
      if (passenger.berthPreference) {
        await selectBerthPreference(i, passenger.berthPreference);
        await wait(50);
      }

      // Select nationality (if field exists)
      if (passenger.nationality) {
        await selectNationality(i, passenger.nationality);
        await wait(50);
      }

      log(`✅ Passenger ${i + 1} details filled`, "success");
    }

    log("All passenger details filled!", "success");
  } catch (error) {
    log(`Error filling passenger details: ${error.message}`, "error");
    throw error;
  }
}

async function fillPassengerName(index, name) {
  try {
    // Find all name input fields
    const nameInputs = document.querySelectorAll('input[placeholder="Name"]');

    if (nameInputs[index]) {
      triggerInputEvent(nameInputs[index], name);

      // Wait for autocomplete and dismiss it
      await wait(150);

      // Press Escape to close autocomplete if it appears
      nameInputs[index].dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Escape",
          keyCode: 27,
          bubbles: true,
        })
      );

      await wait(50);
      log(`Name filled: ${name}`, "success");
    } else {
      throw new Error(`Name input ${index} not found`);
    }
  } catch (error) {
    log(`Error filling name: ${error.message}`, "error");
    throw error;
  }
}

async function fillPassengerAge(index, age) {
  try {
    const ageInputs = document.querySelectorAll('input[placeholder="Age"]');

    if (ageInputs[index]) {
      triggerInputEvent(ageInputs[index], age.toString());
      log(`Age filled: ${age}`, "success");
    } else {
      throw new Error(`Age input ${index} not found`);
    }
  } catch (error) {
    log(`Error filling age: ${error.message}`, "error");
    throw error;
  }
}

async function selectPassengerGender(index, gender) {
  try {
    const genderSelects = document.querySelectorAll(
      'select[formcontrolname="passengerGender"]'
    );

    if (genderSelects[index]) {
      genderSelects[index].value = gender;
      genderSelects[index].dispatchEvent(
        new Event("change", { bubbles: true })
      );
      log(`Gender selected: ${gender}`, "success");
    } else {
      throw new Error(`Gender select ${index} not found`);
    }
  } catch (error) {
    log(`Error selecting gender: ${error.message}`, "error");
    throw error;
  }
}

async function selectBerthPreference(index, berth) {
  try {
    const berthSelects = document.querySelectorAll(
      'select[formcontrolname="passengerBerthChoice"]'
    );

    if (berthSelects[index]) {
      berthSelects[index].value = berth;
      berthSelects[index].dispatchEvent(new Event("change", { bubbles: true }));
      log(`Berth preference selected: ${berth}`, "success");
    } else {
      log(`Berth select ${index} not found, skipping`, "warning");
    }
  } catch (error) {
    log(`Error selecting berth: ${error.message}`, "warning");
  }
}

async function selectNationality(index, nationality) {
  try {
    // Nationality dropdown might be different, adjust selector as needed
    const nationalitySelects = document.querySelectorAll(
      'select[formcontrolname="passengerNationality"]'
    );

    if (nationalitySelects[index]) {
      nationalitySelects[index].value = nationality;
      nationalitySelects[index].dispatchEvent(
        new Event("change", { bubbles: true })
      );
      log(`Nationality selected: ${nationality}`, "success");
    }
  } catch (error) {
    // Nationality might not always be present, so just log warning
    log(`Nationality field not found or error: ${error.message}`, "warning");
  }
}

async function setBookingPreferences(autoUpgrade, confirmBerthsOnly) {
  log("Setting booking preferences...", "info");

  try {
    // Auto Upgradation checkbox
    if (autoUpgrade) {
      const autoUpgradeCheckbox = document.querySelector(
        'input[id="autoUpgradation"]'
      );
      if (autoUpgradeCheckbox && !autoUpgradeCheckbox.checked) {
        autoUpgradeCheckbox.click();
        log("Auto upgradation enabled", "success");
      }
    }

    // Confirm berths only checkbox
    if (confirmBerthsOnly) {
      const confirmBerthsCheckbox = document.querySelector(
        'input[id="confirmberths"]'
      );
      if (confirmBerthsCheckbox && !confirmBerthsCheckbox.checked) {
        confirmBerthsCheckbox.click();
        log("Confirm berths only enabled", "success");
      }
    }

    await wait(100);
  } catch (error) {
    log(`Error setting preferences: ${error.message}`, "warning");
  }
}

async function selectPaymentMethod(paymentType, upiId = null) {
  log(`Setting up payment method: ${paymentType}...`, "info");

  try {
    // STEP 1: Make sure "Pay with Loyalty Points" is NOT selected (always FALSE)
    const loyaltyRadio = document.querySelector(
      'input[name="loyalityOperationType"][value="2"]'
    );
    if (loyaltyRadio) {
      // Check if it's selected
      if (loyaltyRadio.checked) {
        log("Loyalty Points was selected, deselecting...", "warning");
        // Click the radio button container to deselect
        const loyaltyContainer = loyaltyRadio.closest("p-radiobutton");
        if (loyaltyContainer) {
          loyaltyContainer.click();
        }
      }
      log("Loyalty Points: DISABLED ✓", "success");
    }

    await wait(100);

    // STEP 2: Select UPI Payment (always TRUE for your case)
    if (paymentType === "UPI") {
      // Find the UPI radio button by name and value
      const upiRadio = document.querySelector(
        'input[name="paymentType"][value="2"]'
      );

      if (upiRadio) {
        // Check if already selected
        if (!upiRadio.checked) {
          log("Clicking UPI payment option...", "click");

          // Method 1: Click the actual radio input
          upiRadio.click();

          // Method 2: Also click the parent p-radiobutton component (backup)
          await wait(50);
          const upiContainer = upiRadio.closest("p-radiobutton");
          if (upiContainer && !upiRadio.checked) {
            const radioBox = upiContainer.querySelector(".ui-radiobutton-box");
            if (radioBox) {
              radioBox.click();
            }
          }

          await wait(100);

          // Verify selection
          if (upiRadio.checked) {
            log("UPI payment selected ✓", "success");
          } else {
            log("UPI selection might have failed, retrying...", "warning");
            // Force click one more time
            const labelElement = document.querySelector('label[for="2"]');
            if (labelElement && labelElement.textContent.includes("BHIM/UPI")) {
              labelElement.click();
              await wait(100);
            }
          }
        } else {
          log("UPI payment already selected ✓", "success");
        }

        // STEP 3: Fill UPI ID if provided
        if (upiId) {
          await wait(100);
          log(`Filling UPI ID: ${upiId}...`, "info");

          // Look for UPI input field (might appear after selecting UPI)
          const upiInputSelectors = [
            'input[placeholder*="UPI"]',
            'input[placeholder*="upi"]',
            'input[formcontrolname*="upi"]',
            'input[formcontrolname="vpaId"]',
            'input[name*="upi"]',
            'input[id*="upi"]',
          ];

          let upiInput = null;
          for (let selector of upiInputSelectors) {
            upiInput = document.querySelector(selector);
            if (upiInput) {
              log(
                `Found UPI input field with selector: ${selector}`,
                "success"
              );
              break;
            }
          }

          if (upiInput) {
            triggerInputEvent(upiInput, upiId);
            log(`UPI ID filled: ${upiId} ✓`, "success");
          } else {
            log("UPI ID field not found (might appear later)", "warning");
          }
        }
      } else {
        throw new Error("UPI radio button not found");
      }
    }

    await wait(100);
    log("Payment method setup complete!", "success");
  } catch (error) {
    log(`Error setting payment method: ${error.message}`, "error");
    throw error;
  }
}

async function clickContinueToPayment() {
  log("Looking for Continue button...", "search");

  try {
    await wait(100);

    // Method 1: Find the specific button with exact classes
    let continueButton = document.querySelector(
      'button[type="submit"].train_Search.btnDefault'
    );

    // Method 2: Backup - search by text content
    if (!continueButton) {
      log("Trying alternate selector...", "search");
      const buttons = document.querySelectorAll(
        "button.train_Search.btnDefault"
      );

      for (let button of buttons) {
        const buttonText = button.textContent.trim();
        if (buttonText === "Continue" || buttonText === "CONTINUE") {
          continueButton = button;
          log("Found Continue button by text match", "success");
          break;
        }
      }
    }

    // Method 3: Final fallback - any submit button with Continue text
    if (!continueButton) {
      log("Trying final fallback...", "search");
      const allButtons = document.querySelectorAll('button[type="submit"]');

      for (let button of allButtons) {
        if (button.textContent.trim().includes("Continue")) {
          continueButton = button;
          log("Found Continue button (fallback method)", "success");
          break;
        }
      }
    }

    if (!continueButton) {
      throw new Error("Continue button not found");
    }

    // Check if button is disabled
    if (
      continueButton.disabled ||
      continueButton.classList.contains("disabled")
    ) {
      log("⚠️  Continue button is disabled", "warning");
      log(
        "Please check if all required fields are filled correctly",
        "warning"
      );
      log("Attempting to click anyway...", "info");
    }

    // Scroll into view
    log("Scrolling to Continue button...", "info");
    continueButton.scrollIntoView({ behavior: "smooth", block: "center" });
    await wait(100);

    // Click the button
    log("🎯 Clicking Continue button...", "click");
    continueButton.click();

    await wait(100);

    // Verify if page changed or form submitted
    const currentUrl = window.location.href;
    log(`Current URL: ${currentUrl}`, "info");

    await wait(100);

    const newUrl = window.location.href;
    if (newUrl !== currentUrl) {
      log("✅ Page navigated successfully!", "success");
      log(`New URL: ${newUrl}`, "success");
    } else {
      log("⏳ Waiting for page response...", "wait");
      await wait(150);

      // Check one more time
      const finalUrl = window.location.href;
      if (finalUrl !== currentUrl) {
        log("✅ Navigation completed!", "success");
      } else {
        log(
          "⚠️  Page did not change - there might be validation errors",
          "warning"
        );

        // Check for error messages
        const errorMessages = document.querySelectorAll(
          ".error-message, .alert-danger, .text-danger, .error"
        );
        if (errorMessages.length > 0) {
          log("Found error messages on page:", "error");
          errorMessages.forEach((error, index) => {
            log(`Error ${index + 1}: ${error.textContent.trim()}`, "error");
          });
        }
      }
    }

    log("✅ Continue button clicked successfully!", "success");
    return true;
  } catch (error) {
    log(`❌ Error clicking continue: ${error.message}`, "error");
    throw error;
  }
}

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

    await wait(100);

    // STEP 3: Fill Journey Date
    await fillDateField(TICKET_CONFIG.TRAVEL_DATE);

    await wait(100);

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

    await wait(100);

    // STEP 6: Click Search Button
    const searchBtn = await waitForElement("button.search_btn", 200);
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
    await wait(300);

    // STEP 7: Find the specific train
    const trainElement = await findTrainByNumber(
      TICKET_CONFIG.TARGET_TRAIN_NUMBER
    );

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
      log(
        `Selected: ${dateSelection.date} - Status: ${dateSelection.status}`,
        "success"
      );
    }

    // STEP 10: Click Book Now
    await clickBookNow(trainElement);

    console.log("═".repeat(50));
    log("BOOKING FLOW COMPLETED SUCCESSFULLY!", "success");
    log("Proceeding to passenger details page...", "info");
    log("BOOKING FLOW COMPLETED! Now filling passenger details...", "success");
    console.log("═".repeat(50));
    // Wait for passenger form page to load
    await wait(200);

    // STEP 11: Fill passenger details
    await fillPassengerDetails(TICKET_CONFIG.PASSENGERS);

    await wait(200);

    // STEP 12: Set booking preferences
    await setBookingPreferences(
      TICKET_CONFIG.AUTO_UPGRADATION,
      TICKET_CONFIG.CONFIRM_BERTHS_ONLY
    );

    await wait(200);

    // STEP 13: Select payment method
    await selectPaymentMethod(TICKET_CONFIG.PAYMENT_TYPE, TICKET_CONFIG.UPI_ID);

    await wait(500);

    // STEP 14: Click continue to payment
    await clickContinueToPayment();

    console.log("═".repeat(50));
    log("PASSENGER FORM COMPLETED SUCCESSFULLY!", "success");
    log("Ready for payment/captcha...", "info");
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
  handlePopup: handleDatePopup,
};

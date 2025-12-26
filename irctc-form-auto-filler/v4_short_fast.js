// ==========================================
// IRCTC OPTIMIZED AUTOFILL SCRIPT
// ==========================================

const TICKET_CONFIG = {
  FROM_STATION_CODE: "GZB",
  FROM_STATION_FULL: "GHAZIABAD - GZB",
  TO_STATION_CODE: "HRI",
  TO_STATION_FULL: "HARDOI - HRI",
  TRAVEL_DATE: "20/12/2025",
  TRAIN_CLASS: "All Classes",
  QUOTA: "PREMIUM TATKAL",
  TARGET_TRAIN_NUMBER: "12230",
  PREFERRED_CLASS: "3A",
  ALTERNATIVE_CLASSES: ["SL", "3E", "2A", "1A"],
  PASSENGERS: [
    {
      name: "Ayush Chowdhary",
      age: 30,
      gender: "M",
      berthPreference: "LB",
      foodChoice: "V",
      nationality: "IN",
    },
  ],
  AUTO_UPGRADATION: true,
  CONFIRM_BERTHS_ONLY: true,
  PAYMENT_TYPE: "UPI",
  USE_LOYALTY_POINTS: false,
  UPI_ID: "yourname@paytm",
};

// ==========================================
// CORE UTILITIES
// ==========================================

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

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

function triggerInputEvent(element, value) {
  element.focus();
  element.value = value;
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
  element.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true }));
}

// ==========================================
// FORM FILLING FUNCTIONS
// ==========================================

async function fillStation(inputSelector, stationCode, stationFullMatch) {
  const input = await waitForElement(inputSelector, 2000);
  triggerInputEvent(input, stationCode);
  await wait(200);

  try {
    await waitForElement(".ui-autocomplete-items li", 2000);
  } catch {
    triggerInputEvent(input, stationCode + " ");
    await waitForElement(".ui-autocomplete-items li", 2000);
  }

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
  dateInput.click();
  await wait(300);

  await waitForElement(".ui-datepicker", 2000);

  const currentMonth = document.querySelector(".ui-datepicker-month");
  const currentYear = document.querySelector(".ui-datepicker-year");

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

    if (currMonth === targetMonthName && currYear === targetYear) break;

    const currDate = new Date(
      parseInt(currYear),
      monthNames.indexOf(currMonth),
      1
    );
    const targetDate = new Date(year, month - 1, 1);

    if (targetDate > currDate) {
      const nextBtn = document.querySelector(".ui-datepicker-next");
      if (nextBtn && !nextBtn.classList.contains("ui-state-disabled"))
        nextBtn.click();
    } else {
      const prevBtn = document.querySelector(".ui-datepicker-prev");
      if (prevBtn && !prevBtn.classList.contains("ui-state-disabled"))
        prevBtn.click();
    }

    await wait(100);
    attempts++;
  }

  const dayLinks = document.querySelectorAll(
    ".ui-datepicker-calendar a.ui-state-default"
  );
  for (let dayLink of dayLinks) {
    if (parseInt(dayLink.textContent.trim()) === day) {
      dayLink.click();
      await wait(150);
      return;
    }
  }
}

async function selectDropdown(dropdownSelector, optionText) {
  try {
    const dropdown = await waitForElement(dropdownSelector, 200);
    dropdown.click();
    await wait(200);

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
// TRAIN SEARCH & BOOKING
// ==========================================

async function findTrainByNumber(trainNumber) {
  await wait(1500);
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

async function selectTrainClass(
  trainElement,
  preferredClass,
  alternativeClasses = []
) {
  const classesToTry = [preferredClass, ...alternativeClasses];

  for (let classCode of classesToTry) {
    const className = CLASS_MAP[classCode];
    if (!className) continue;

    const classSpans = trainElement.querySelectorAll("span.hidden-xs");
    for (let span of classSpans) {
      if (span.textContent.trim() === className) {
        span.click();
        await wait(150);
        return { success: true, className, classCode };
      }
    }

    const classButtons = trainElement.querySelectorAll(".pre-avl");
    for (let button of classButtons) {
      const buttonText = button.querySelector("strong");
      if (buttonText && buttonText.textContent.includes(className)) {
        button.click();
        await wait(150);
        return { success: true, className, classCode };
      }
    }
  }
  throw new Error("No available class found");
}

async function handleDatePopup(targetDate) {
  await wait(150);
  const dateOptions = document.querySelectorAll(".pre-avl");

  const [day, month] = targetDate.split("/");
  const targetDay = parseInt(day);
  const monthNames = [
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
    selectedOption.element.classList.add("selected-class");
    selectedOption.element.click();
    await wait(1000);
    return { date: selectedOption.dateText };
  }
  throw new Error("No date option found");
}

async function clickBookNow(trainElement) {
  await wait(200);

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
      if (
        button.textContent.includes("Book Now") &&
        button.classList.contains("train_Search")
      ) {
        bookButton = button;
        break;
      }
    }
  }

  if (!bookButton) throw new Error("Book Now button not found");

  bookButton.scrollIntoView({ behavior: "smooth", block: "center" });
  await wait(200);
  bookButton.click();
  await wait(250);
}

// ==========================================
// PASSENGER FORM FILLING
// ==========================================

async function fillPassengerDetails(passengers) {
  await wait(1500);

  for (let i = 0; i < passengers.length; i++) {
    const p = passengers[i];

    // Name
    const nameInputs = document.querySelectorAll('input[placeholder="Name"]');
    if (nameInputs[i]) {
      triggerInputEvent(nameInputs[i], p.name);
      await wait(150);
      nameInputs[i].dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Escape",
          keyCode: 27,
          bubbles: true,
        })
      );
      await wait(50);
    }

    // Age
    const ageInputs = document.querySelectorAll('input[placeholder="Age"]');
    if (ageInputs[i]) {
      triggerInputEvent(ageInputs[i], p.age.toString());
      await wait(50);
    }

    // Gender
    const genderSelects = document.querySelectorAll(
      'select[formcontrolname="passengerGender"]'
    );
    if (genderSelects[i]) {
      genderSelects[i].value = p.gender;
      genderSelects[i].dispatchEvent(new Event("change", { bubbles: true }));
      await wait(50);
    }

    // Berth
    if (p.berthPreference) {
      const berthSelects = document.querySelectorAll(
        'select[formcontrolname="passengerBerthChoice"]'
      );
      if (berthSelects[i]) {
        berthSelects[i].value = p.berthPreference;
        berthSelects[i].dispatchEvent(new Event("change", { bubbles: true }));
        await wait(50);
      }
    }

    // Nationality
    if (p.nationality) {
      try {
        const nationalitySelects = document.querySelectorAll(
          'select[formcontrolname="passengerNationality"]'
        );
        if (nationalitySelects[i]) {
          nationalitySelects[i].value = p.nationality;
          nationalitySelects[i].dispatchEvent(
            new Event("change", { bubbles: true })
          );
        }
      } catch {}
    }
  }
}

async function setBookingPreferences(autoUpgrade, confirmBerthsOnly) {
  try {
    if (autoUpgrade) {
      const autoUpgradeCheckbox = document.querySelector(
        'input[id="autoUpgradation"]'
      );
      if (autoUpgradeCheckbox && !autoUpgradeCheckbox.checked) {
        autoUpgradeCheckbox.click();
      }
    }

    if (confirmBerthsOnly) {
      const confirmBerthsCheckbox = document.querySelector(
        'input[id="confirmberths"]'
      );
      if (confirmBerthsCheckbox && !confirmBerthsCheckbox.checked) {
        confirmBerthsCheckbox.click();
      }
    }
  } catch {}
}

async function selectPaymentMethod(paymentType, upiId = null) {
  try {
    // Disable loyalty points
    const loyaltyRadio = document.querySelector(
      'input[name="loyalityOperationType"][value="2"]'
    );
    if (loyaltyRadio && loyaltyRadio.checked) {
      const loyaltyContainer = loyaltyRadio.closest("p-radiobutton");
      if (loyaltyContainer) loyaltyContainer.click();
    }

    await wait(100);

    // Select UPI
    if (paymentType === "UPI") {
      const upiRadio = document.querySelector(
        'input[name="paymentType"][value="2"]'
      );
      if (upiRadio && !upiRadio.checked) {
        upiRadio.click();
        await wait(50);

        const upiContainer = upiRadio.closest("p-radiobutton");
        if (upiContainer && !upiRadio.checked) {
          const radioBox = upiContainer.querySelector(".ui-radiobutton-box");
          if (radioBox) radioBox.click();
        }

        await wait(100);

        // Fill UPI ID
        if (upiId) {
          const upiInputSelectors = [
            'input[placeholder*="UPI"]',
            'input[formcontrolname="vpaId"]',
            'input[name*="upi"]',
          ];

          for (let selector of upiInputSelectors) {
            const upiInput = document.querySelector(selector);
            if (upiInput) {
              triggerInputEvent(upiInput, upiId);
              break;
            }
          }
        }
      }
    }
  } catch {}
}

async function clickContinueToPayment() {
  await wait(100);

  let continueButton = document.querySelector(
    'button[type="submit"].train_Search.btnDefault'
  );

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
  await wait(100);
  continueButton.click();
  await wait(1500);
}

// ==========================================
// MAIN EXECUTION
// ==========================================

async function fillIRCTCForm() {
  try {
    // Step 1-2: Fill stations
    await fillStation(
      'input[aria-label="Enter From station. Input is Mandatory."]',
      TICKET_CONFIG.FROM_STATION_CODE,
      TICKET_CONFIG.FROM_STATION_FULL
    );
    await wait(150);

    await fillStation(
      'input[aria-label="Enter To station. Input is Mandatory."]',
      TICKET_CONFIG.TO_STATION_CODE,
      TICKET_CONFIG.TO_STATION_FULL
    );
    await wait(100);

    // Step 3: Fill date
    await fillDateField(TICKET_CONFIG.TRAVEL_DATE);
    await wait(100);

    // Step 4-5: Class & Quota
    if (
      TICKET_CONFIG.TRAIN_CLASS &&
      TICKET_CONFIG.TRAIN_CLASS !== "All Classes"
    ) {
      await selectDropdown(
        ".ng-tns-c76-10.ui-dropdown",
        TICKET_CONFIG.TRAIN_CLASS
      );
    }

    if (TICKET_CONFIG.QUOTA && TICKET_CONFIG.QUOTA !== "GENERAL") {
      await selectDropdown(".ng-tns-c76-11.ui-dropdown", TICKET_CONFIG.QUOTA);
    }
    await wait(100);

    // Step 6: Search
    const searchBtn = await waitForElement("button.search_btn", 200);
    searchBtn.click();
    await wait(300);

    // Step 7-10: Train selection & booking
    const trainElement = await findTrainByNumber(
      TICKET_CONFIG.TARGET_TRAIN_NUMBER
    );
    await selectTrainClass(
      trainElement,
      TICKET_CONFIG.PREFERRED_CLASS,
      TICKET_CONFIG.ALTERNATIVE_CLASSES
    );
    await handleDatePopup(TICKET_CONFIG.TRAVEL_DATE);
    await clickBookNow(trainElement);

    // Step 11-14: Passenger & payment
    await fillPassengerDetails(TICKET_CONFIG.PASSENGERS);
    await wait(200);
    await setBookingPreferences(
      TICKET_CONFIG.AUTO_UPGRADATION,
      TICKET_CONFIG.CONFIRM_BERTHS_ONLY
    );
    await wait(200);
    await selectPaymentMethod(TICKET_CONFIG.PAYMENT_TYPE, TICKET_CONFIG.UPI_ID);
    await wait(500);
    await clickContinueToPayment();
  } catch (err) {
    console.error("Autofill failed:", err.message);
  }
}

// Auto-run
fillIRCTCForm();
window.irctcAutofill = fillIRCTCForm;

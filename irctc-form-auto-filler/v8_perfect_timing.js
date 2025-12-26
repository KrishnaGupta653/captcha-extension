// ==========================================
// IRCTC TATKAL ROBUST & FAST SCRIPT
// Fixed station filling + Better error handling
// ==========================================

const TICKET_CONFIG = {
  FROM_STATION_CODE: "GZB",
  FROM_STATION_FULL: "GHAZIABAD - GZB",
  TO_STATION_CODE: "HRI",
  TO_STATION_FULL: "HARDOI - HRI",
  TRAVEL_DATE: "25/12/2025",
  TRAIN_CLASS: "All Classes",
  QUOTA: "PREMIUM TATKAL",
  TARGET_TRAIN_NUMBER: "12230",
  PREFERRED_CLASS: "2A",
  ALTERNATIVE_CLASSES: ["3A", "2A", "1A"],
  PASSENGERS: [
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
};

// ==========================================
// TIMING CONFIGURATION
// ==========================================
const TIMING = {
  afterStation: () => 80 + Math.random() * 60,       // 80-140ms
  afterAutocomplete: () => 150 + Math.random() * 100, // 150-250ms
  afterDate: () => 60 + Math.random() * 40,          // 60-100ms
  afterDropdown: () => 80 + Math.random() * 40,      // 80-120ms
  afterSearch: () => 1000 + Math.random() * 500,     // 1000-1500ms
  afterClassSelect: () => 120 + Math.random() * 80,  // 120-200ms
  beforeBook: () => 200 + Math.random() * 150,       // 200-350ms
  afterBook: () => 700 + Math.random() * 300,        // 700-1000ms
  betweenFields: () => 50 + Math.random() * 40,      // 50-90ms
  typing: () => 70 + Math.random() * 50,             // 70-120ms per char
  shortWait: () => 30 + Math.random() * 20,          // 30-50ms
  polling: () => 40 + Math.random() * 30,            // 40-70ms
};

const wait = (ms) => new Promise(r => setTimeout(r, ms));
const rWait = (fn) => wait(fn());

function log(msg, type = 'info') {
  const time = new Date().toISOString().substr(11, 8);
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warn' ? '⚠️' : 'ℹ️';
  console.log(`[${time}] ${icon} ${msg}`);
}

// ==========================================
// ROBUST ELEMENT WAITING
// ==========================================
async function waitEl(selector, options = {}) {
  const {
    timeout = 5000,
    visible = true,
    enabled = true,
    description = selector
  } = options;
  
  log(`Waiting for: ${description}`);
  const start = Date.now();
  
  while (Date.now() - start < timeout) {
    const el = document.querySelector(selector);
    
    if (el) {
      const isVisible = !visible || (el.offsetHeight > 0 && el.offsetWidth > 0);
      const isEnabled = !enabled || (!el.disabled && !el.hasAttribute('disabled'));
      const style = window.getComputedStyle(el);
      const notHidden = style.display !== 'none' && style.visibility !== 'hidden';
      
      if (isVisible && isEnabled && notHidden) {
        log(`Found: ${description}`, 'success');
        return el;
      }
    }
    
    await wait(TIMING.polling());
  }
  
  throw new Error(`Timeout: ${description}`);
}

async function waitForAny(selectors, timeout = 5000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    for (let sel of selectors) {
      const el = document.querySelector(sel);
      if (el && el.offsetHeight > 0) return el;
    }
    await wait(50);
  }
  return null;
}

// ==========================================
// INPUT HELPERS
// ==========================================
function triggerEvents(el, value) {
  // Focus first
  el.focus();
  
  // Set value
  if (value !== undefined) {
    el.value = value;
  }
  
  // Trigger all necessary events for Angular
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
  el.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));
  
  // Mark as touched
  el.dispatchEvent(new Event('blur', { bubbles: true }));
}

async function typeText(el, text, description = 'text') {
  log(`Typing: ${description}`);
  el.focus();
  el.value = '';
  
  for (let char of text) {
    el.value += char;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new KeyboardEvent('keydown', { key: char, bubbles: true }));
    await rWait(TIMING.typing);
  }
  
  el.dispatchEvent(new Event('change', { bubbles: true }));
  el.blur();
  log(`Typed: ${description}`, 'success');
}

function clickElement(el, description = 'element') {
  log(`Clicking: ${description}`);
  
  const rect = el.getBoundingClientRect();
  const x = rect.left + rect.width * (0.4 + Math.random() * 0.2);
  const y = rect.top + rect.height * (0.4 + Math.random() * 0.2);
  
  // Dispatch mouse events
  el.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true, clientX: x, clientY: y }));
  el.dispatchEvent(new MouseEvent('mouseover', { bubbles: true, clientX: x, clientY: y }));
  el.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, clientX: x, clientY: y, button: 0 }));
  el.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, clientX: x, clientY: y, button: 0 }));
  el.click();
  el.dispatchEvent(new MouseEvent('mouseleave', { bubbles: true }));
  
  log(`Clicked: ${description}`, 'success');
}

// ==========================================
// ERROR DIALOG MONITOR
// ==========================================
let errorMonitor = null;

function startErrorMonitor() {
  if (errorMonitor) return;
  
  log('Starting error monitor');
  errorMonitor = setInterval(() => {
    const dialogs = document.querySelectorAll('.ui-dialog, .p-dialog, [role="dialog"]');
    
    for (let dialog of dialogs) {
      const text = dialog.textContent || '';
      if (text.match(/error|failed|process|invalid/i)) {
        log('Found error dialog, closing...', 'warn');
        const closeBtn = dialog.querySelector(
          'button[aria-label="Close"], .ui-dialog-titlebar-close, button.ui-dialog-titlebar-icon'
        );
        if (closeBtn) {
          closeBtn.click();
        }
      }
    }
  }, 150);
}

function stopErrorMonitor() {
  if (errorMonitor) {
    clearInterval(errorMonitor);
    errorMonitor = null;
    log('Stopped error monitor');
  }
}

// ==========================================
// STATION FILLING (ROBUST)
// ==========================================
async function fillStation(selector, code, fullName, description) {
  log(`Filling station: ${description}`);
  
  try {
    // Step 1: Find and focus input
    const input = await waitEl(selector, {
      timeout: 3000,
      description: `${description} input`
    });
    
    // Step 2: Clear any existing value
    input.value = '';
    input.focus();
    await rWait(TIMING.shortWait);
    
    // Step 3: Type station code with events
    for (let char of code) {
      input.value += char;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new KeyboardEvent('keydown', { key: char, bubbles: true }));
      await rWait(TIMING.typing);
    }
    
    // Trigger keyup to ensure autocomplete
    input.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));
    
    log(`Waiting for autocomplete list...`);
    
    // Step 4: Wait for autocomplete with multiple attempts
    let autocompleteList = null;
    let attempts = 0;
    
    while (attempts < 3 && !autocompleteList) {
      await wait(200 + attempts * 100); // 200ms, 300ms, 400ms
      
      autocompleteList = document.querySelector('.ui-autocomplete-items');
      
      if (!autocompleteList || autocompleteList.children.length === 0) {
        log(`Attempt ${attempts + 1}: Autocomplete not shown, retrying...`, 'warn');
        
        // Try adding space to trigger
        input.value = code + ' ';
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));
        
        attempts++;
      } else {
        break;
      }
    }
    
    if (!autocompleteList || autocompleteList.children.length === 0) {
      throw new Error('Autocomplete list not appearing');
    }
    
    await rWait(TIMING.afterAutocomplete);
    
    // Step 5: Find and click matching station
    const items = document.querySelectorAll('.ui-autocomplete-items li');
    log(`Found ${items.length} stations in autocomplete`);
    
    let clicked = false;
    
    // Try exact match first
    for (let item of items) {
      const text = item.innerText.trim();
      if (text.includes(fullName) || text.includes(code)) {
        log(`Selecting: ${text.substring(0, 50)}...`);
        clickElement(item, `Station: ${code}`);
        clicked = true;
        break;
      }
    }
    
    // Fallback to first item
    if (!clicked && items.length > 0) {
      log(`Exact match not found, selecting first option`, 'warn');
      clickElement(items[0], 'First station');
      clicked = true;
    }
    
    if (!clicked) {
      throw new Error('No station items to click');
    }
    
    // Step 6: Verify selection
    await wait(100);
    if (input.value.trim().length > 0) {
      log(`Station ${description} filled successfully`, 'success');
    } else {
      log(`Warning: Station value may not be set`, 'warn');
    }
    
    await rWait(TIMING.afterStation);
    
  } catch (err) {
    log(`Error filling station ${description}: ${err.message}`, 'error');
    throw err;
  }
}

// ==========================================
// DATE FILLING (ROBUST)
// ==========================================
async function fillDate(dateStr, description = 'date') {
  log(`Filling date: ${dateStr}`);
  
  try {
    const [day, month, year] = dateStr.split('/').map(Number);
    
    // Step 1: Find and click date input
    const dateInput = await waitEl('.ui-calendar input', {
      timeout: 3000,
      description: 'Date input'
    });
    
    clickElement(dateInput, 'Date picker');
    await wait(200 + Math.random() * 150);
    
    // Step 2: Wait for calendar
    const calendar = await waitEl('.ui-datepicker', {
      timeout: 2000,
      description: 'Date picker calendar'
    });
    
    // Step 3: Navigate to correct month/year
    const monthNames = [
      'January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December'
    ];
    const targetMonth = monthNames[month - 1];
    const targetYear = year.toString();
    
    let attempts = 0;
    while (attempts < 24) {
      const currMonthEl = document.querySelector('.ui-datepicker-month');
      const currYearEl = document.querySelector('.ui-datepicker-year');
      
      if (!currMonthEl || !currYearEl) break;
      
      const currMonth = currMonthEl.textContent.trim();
      const currYear = currYearEl.textContent.trim();
      
      if (currMonth === targetMonth && currYear === targetYear) {
        log(`Reached target month: ${targetMonth} ${targetYear}`, 'success');
        break;
      }
      
      // Calculate direction
      const curr = new Date(parseInt(currYear), monthNames.indexOf(currMonth), 1);
      const target = new Date(year, month - 1, 1);
      
      const btn = target > curr ?
        document.querySelector('.ui-datepicker-next') :
        document.querySelector('.ui-datepicker-prev');
      
      if (btn && !btn.classList.contains('ui-state-disabled')) {
        clickElement(btn, target > curr ? 'Next month' : 'Previous month');
        await wait(80 + Math.random() * 40);
      } else {
        break;
      }
      
      attempts++;
    }
    
    // Step 4: Click the day
    await wait(100 + Math.random() * 50);
    const dayLinks = document.querySelectorAll('.ui-datepicker-calendar a.ui-state-default');
    
    let dayClicked = false;
    for (let link of dayLinks) {
      if (parseInt(link.textContent.trim()) === day) {
        clickElement(link, `Day ${day}`);
        dayClicked = true;
        break;
      }
    }
    
    if (!dayClicked) {
      throw new Error(`Day ${day} not found in calendar`);
    }
    
    log(`Date filled: ${dateStr}`, 'success');
    await rWait(TIMING.afterDate);
    
  } catch (err) {
    log(`Error filling date: ${err.message}`, 'error');
    throw err;
  }
}

// ==========================================
// DROPDOWN SELECTION (ROBUST)
// ==========================================
async function selectDropdown(selector, optionText, description) {
  log(`Selecting dropdown: ${description} = ${optionText}`);
  
  try {
    const dropdown = await waitEl(selector, {
      timeout: 2000,
      description: `Dropdown: ${description}`
    });
    
    clickElement(dropdown, description);
    await wait(150 + Math.random() * 100);
    
    // Wait for options
    const optionsList = await waitEl('.ui-dropdown-items, p-dropdownitem', {
      timeout: 2000,
      description: 'Dropdown options'
    });
    
    await wait(50 + Math.random() * 30);
    
    // Find and click option
    const options = document.querySelectorAll('.ui-dropdown-items li, p-dropdownitem li');
    
    for (let opt of options) {
      const text = opt.innerText.trim();
      if (text === optionText || text.includes(optionText)) {
        clickElement(opt, `Option: ${optionText}`);
        log(`Selected: ${optionText}`, 'success');
        return true;
      }
    }
    
    log(`Option "${optionText}" not found`, 'warn');
    return false;
    
  } catch (err) {
    log(`Dropdown error: ${err.message}`, 'warn');
    return false;
  }
}

// ==========================================
// SEARCH WITH RETRY
// ==========================================
async function searchTrains() {
  log('Searching for trains...');
  
  const start = Date.now();
  const maxTime = 90000; // 90 seconds
  
  while (Date.now() - start < maxTime) {
    try {
      // Check if already on train list page
      const trainList = document.querySelector('app-train-avl-enq');
      if (trainList) {
        log('Already on train list page', 'success');
        return true;
      }
      
      // Find and click search button
      const searchBtn = document.querySelector('button.search_btn');
      if (searchBtn && !searchBtn.disabled) {
        clickElement(searchBtn, 'Search button');
        
        // Wait for results
        await wait(300 + Math.random() * 200);
        
        // Check if trains appeared
        const trains = document.querySelector('app-train-avl-enq');
        if (trains) {
          log('Trains loaded!', 'success');
          return true;
        }
      }
      
      await wait(250 + Math.random() * 150);
      
    } catch (err) {
      log(`Search retry: ${err.message}`, 'warn');
      await wait(500);
    }
  }
  
  throw new Error('Search failed after timeout');
}

// ==========================================
// TRAIN SELECTION (ROBUST)
// ==========================================
async function findTrain(trainNumber) {
  log(`Looking for train: ${trainNumber}`);
  
  await rWait(TIMING.afterSearch);
  
  const trains = document.querySelectorAll('app-train-avl-enq');
  log(`Found ${trains.length} trains`);
  
  for (let train of trains) {
    const heading = train.querySelector('.train-heading strong');
    if (heading && heading.textContent.includes(trainNumber)) {
      log(`Found train ${trainNumber}!`, 'success');
      return train;
    }
  }
  
  throw new Error(`Train ${trainNumber} not found`);
}

async function selectClass(trainEl, preferred, alternatives) {
  log(`Selecting class: ${preferred}`);
  
  const classMap = {
    'SL': 'Sleeper (SL)',
    '3E': 'AC 3 Economy (3E)',
    '3A': 'AC 3 Tier (3A)',
    '2A': 'AC 2 Tier (2A)',
    '1A': 'AC First Class (1A)'
  };
  
  const classesToTry = [preferred, ...alternatives];
  
  for (let code of classesToTry) {
    const name = classMap[code];
    if (!name) continue;
    
    log(`Trying class: ${name}`);
    
    // Try hidden-xs spans
    const spans = trainEl.querySelectorAll('span.hidden-xs');
    for (let span of spans) {
      if (span.textContent.trim() === name) {
        clickElement(span, `Class: ${name}`);
        await rWait(TIMING.afterClassSelect);
        return code;
      }
    }
    
    // Try pre-avl buttons
    const btns = trainEl.querySelectorAll('.pre-avl');
    for (let btn of btns) {
      const txt = btn.querySelector('strong')?.textContent;
      if (txt && txt.includes(name)) {
        clickElement(btn, `Class: ${name}`);
        await rWait(TIMING.afterClassSelect);
        return code;
      }
    }
  }
  
  throw new Error('No available class found');
}

async function handleDatePopup(targetDate) {
  log('Handling date popup...');
  
  await wait(100 + Math.random() * 50);
  
  const [day, month] = targetDate.split('/').map(Number);
  const monthShort = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][month-1];
  
  const dateOpts = Array.from(document.querySelectorAll('.pre-avl')).filter(opt => {
    const txt = opt.querySelector('strong')?.textContent;
    return txt && /\w+,\s+\d+\s+\w+/.test(txt);
  });
  
  log(`Found ${dateOpts.length} date options`);
  
  for (let opt of dateOpts) {
    const txt = opt.textContent;
    if (txt.includes(`${day} ${monthShort}`)) {
      clickElement(opt, `Date: ${day} ${monthShort}`);
      await wait(400 + Math.random() * 200);
      return;
    }
  }
  
  if (dateOpts.length > 0) {
    log('Target date not found, selecting first option', 'warn');
    clickElement(dateOpts[0], 'First date');
    await wait(400 + Math.random() * 200);
  }
}

// ==========================================
// BOOK WITH RETRY
// ==========================================
async function bookTrain(trainEl) {
  log('Attempting to book train...');
  
  const start = Date.now();
  const maxTime = 90000;
  
  while (Date.now() - start < maxTime) {
    try {
      // Check if already on passenger page
      const passengerPage = document.querySelector('input[placeholder="Name"]');
      if (passengerPage) {
        log('Already on passenger page!', 'success');
        return true;
      }
      
      // Find book button
      let bookBtn = trainEl.querySelector('button.btnDefault.train_Search');
      
      if (!bookBtn) {
        const parent = trainEl.closest('.form-group');
        bookBtn = parent?.querySelector('button.btnDefault.train_Search');
      }
      
      if (!bookBtn) {
        const btns = trainEl.querySelectorAll('button');
        for (let b of btns) {
          if (b.textContent.includes('Book Now') || b.textContent.includes('Book')) {
            bookBtn = b;
            break;
          }
        }
      }
      
      if (bookBtn && !bookBtn.disabled) {
        bookBtn.scrollIntoView({ behavior: 'smooth', block: 'center' });
        await wait(100);
        clickElement(bookBtn, 'Book Now');
        
        await wait(500 + Math.random() * 300);
        
        // Check if passenger page loaded
        const loaded = document.querySelector('input[placeholder="Name"]');
        if (loaded) {
          log('Booking successful!', 'success');
          return true;
        }
      }
      
      await wait(300 + Math.random() * 200);
      
    } catch (err) {
      log(`Book retry: ${err.message}`, 'warn');
      await wait(500);
    }
  }
  
  throw new Error('Booking failed after timeout');
}

// ==========================================
// PASSENGER DETAILS (ROBUST)
// ==========================================
async function fillPassengers(passengers) {
  log('Filling passenger details...');
  
  await rWait(TIMING.afterBook);
  
  for (let i = 0; i < passengers.length; i++) {
    const p = passengers[i];
    log(`Filling passenger ${i + 1}: ${p.name}`);
    
    // Name
    const names = document.querySelectorAll('input[placeholder="Name"]');
    if (names[i]) {
      await typeText(names[i], p.name, `Passenger ${i + 1} name`);
      names[i].dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', keyCode: 27, bubbles: true}));
      await rWait(TIMING.betweenFields);
    }
    
    // Age
    const ages = document.querySelectorAll('input[placeholder="Age"]');
    if (ages[i]) {
      triggerEvents(ages[i], p.age.toString());
      await rWait(TIMING.betweenFields);
    }
    
    // Gender
    const genders = document.querySelectorAll('select[formcontrolname="passengerGender"]');
    if (genders[i]) {
      genders[i].value = p.gender;
      genders[i].dispatchEvent(new Event('change', {bubbles: true}));
      await rWait(TIMING.betweenFields);
    }
    
    // Berth
    if (p.berthPreference) {
      const berths = document.querySelectorAll('select[formcontrolname="passengerBerthChoice"]');
      if (berths[i]) {
        berths[i].value = p.berthPreference;
        berths[i].dispatchEvent(new Event('change', {bubbles: true}));
        await rWait(TIMING.betweenFields);
      }
    }
    
    // Nationality
    if (p.nationality) {
      const nats = document.querySelectorAll('select[formcontrolname="passengerNationality"]');
      if (nats[i]) {
        nats[i].value = p.nationality;
        nats[i].dispatchEvent(new Event('change', {bubbles: true}));
      }
    }
  }
  
  log('Passenger details filled', 'success');
}

async function setPreferences(autoUpgrade, confirmBerths) {
  log('Setting booking preferences...');
  
  if (autoUpgrade) {
    const chk = document.querySelector('input[id="autoUpgradation"]');
    if (chk && !chk.checked) {
      clickElement(chk, 'Auto-upgrade');
    }
  }
  
  if (confirmBerths) {
    const chk = document.querySelector('input[id="confirmberths"]');
    if (chk && !chk.checked) {
      clickElement(chk, 'Confirm berths');
    }
  }
  
  await wait(100 + Math.random() * 50);
}

async function setPayment(type, upiId) {
  log('Setting payment method...');
  
  // Disable loyalty
  try {
    const loyalty = document.querySelector('input[name="loyalityOperationType"][value="2"]');
    if (loyalty?.checked) {
      const container = loyalty.closest('p-radiobutton');
      if (container) clickElement(container, 'Disable loyalty');
    }
  } catch {}
  
  await wait(60 + Math.random() * 40);
  
  if (type === 'UPI') {
    const upi = document.querySelector('input[name="paymentType"][value="2"]');
    if (upi && !upi.checked) {
      clickElement(upi, 'UPI payment');
      await wait(80 + Math.random() * 40);
      
      const container = upi.closest('p-radiobutton');
      if (container && !upi.checked) {
        const box = container.querySelector('.ui-radiobutton-box');
        if (box) clickElement(box, 'UPI radio box');
      }
      
      await wait(100 + Math.random() * 50);
      
      if (upiId) {
        const inp = document.querySelector('input[placeholder*="UPI"], input[formcontrolname="vpaId"]');
        if (inp) {
          triggerEvents(inp, upiId);
          log(`UPI ID set: ${upiId}`, 'success');
        }
      }
    }
  }
  
  await wait(100 + Math.random() * 50);
}

// ==========================================
// MAIN EXECUTION
// ==========================================
async function runIRCTC() {
  const startTime = Date.now();
  
  try {
    log('========================================');
    log('🚀 IRCTC TATKAL SCRIPT STARTED');
    log('========================================');
    
    startErrorMonitor();
    
    // STAGE 1: FORM FILLING
    log('STAGE 1: Filling search form...');
    
    await fillStation(
      'input[aria-label="Enter From station. Input is Mandatory."]',
      TICKET_CONFIG.FROM_STATION_CODE,
      TICKET_CONFIG.FROM_STATION_FULL,
      'FROM station'
    );
    
    await fillStation(
      'input[aria-label="Enter To station. Input is Mandatory."]',
      TICKET_CONFIG.TO_STATION_CODE,
      TICKET_CONFIG.TO_STATION_FULL,
      'TO station'
    );
    
    await fillDate(TICKET_CONFIG.TRAVEL_DATE, 'Journey date');
    
    if (TICKET_CONFIG.TRAIN_CLASS !== 'All Classes') {
      await selectDropdown(
        '.ng-tns-c76-10.ui-dropdown',
        TICKET_CONFIG.TRAIN_CLASS,
        'Class'
      );
    }
    
    if (TICKET_CONFIG.QUOTA !== 'GENERAL') {
      await selectDropdown(
        '.ng-tns-c76-11.ui-dropdown',
        TICKET_CONFIG.QUOTA,
        'Quota'
      );
    }
    
    await rWait(TIMING.afterDropdown);
    
    // STAGE 2: SEARCH
    log('STAGE 2: Searching trains...');
    await searchTrains();
    
    // STAGE 3: TRAIN SELECTION
    log('STAGE 3: Selecting train...');
    const train = await findTrain(TICKET_CONFIG.TARGET_TRAIN_NUMBER);
    
    await selectClass(
      train,
      TICKET_CONFIG.PREFERRED_CLASS,
      TICKET_CONFIG.ALTERNATIVE_CLASSES
    );
    
    await handleDatePopup(TICKET_CONFIG.TRAVEL_DATE);
    await rWait(TIMING.beforeBook);
    
    // STAGE 4: BOOKING
    log('STAGE 4: Booking train...');
    await bookTrain(train);
    
    // STAGE 5: PASSENGERS
    log('STAGE 5: Filling passenger details...');
    await fillPassengers(TICKET_CONFIG.PASSENGERS);
    
    await wait(120 + Math.random() * 80);
    await setPreferences(TICKET_CONFIG.AUTO_UPGRADATION, TICKET_CONFIG.CONFIRM_BERTHS_ONLY);
    
    await wait(100 + Math.random() * 50);
    await setPayment(TICKET_CONFIG.PAYMENT_TYPE, TICKET_CONFIG.UPI_ID);
    
    await wait(200 + Math.random() * 100);
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    
    log('========================================');
    log(`✅ FORM COMPLETED IN ${elapsed}s`, 'success');
    log('Ready for CAPTCHA - Continue manually');
    log('========================================');
    
    stopErrorMonitor();
    
  } catch (err) {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    log('========================================');
    log(`❌ ERROR after ${elapsed}s: ${err.message}`, 'error');
    log('========================================');
    stopErrorMonitor();
    throw err;
  }
}

// ==========================================
// EXPORTS
// ==========================================
window.irctcRun = runIRCTC;
window.irctcStop = stopErrorMonitor;

log('✨ Script loaded successfully');
log('⏱️  Starting in 2 seconds...');
log('🛑 To stop: window.irctcStop()');

setTimeout(runIRCTC, 2000);
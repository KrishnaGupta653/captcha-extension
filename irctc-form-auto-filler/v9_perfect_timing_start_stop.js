

const TICKET_CONFIG = {
  // FROM_STATION_CODE: "GZB",
  // FROM_STATION_FULL: "GHAZIABAD - GZB",
  // TO_STATION_CODE: "HRI",
  // TO_STATION_FULL: "HARDOI - HRI",
  // TRAVEL_DATE: "02/02/2026",
  // TRAIN_CLASS: "All Classes",
  // QUOTA: "GENERAL",
  // TARGET_TRAIN_NUMBER: "12230",
  // PREFERRED_CLASS: "3A",
  // ALTERNATIVE_CLASSES: ["3A", "2A", "1A"],
  // PASSENGERS: [
  FROM_STATION_CODE: "KPD",
  FROM_STATION_FULL: "KATPADI JN - KPD (VELLORE)",
  TO_STATION_CODE: "RNC",
  TO_STATION_FULL: "RANCHI - RNC (HATIA/RANCHI)",
  TRAVEL_DATE: "25/12/2025",
  TRAIN_CLASS: "All Classes",
  QUOTA: "TATKAL",
  TARGET_TRAIN_NUMBER: "13352",
  PREFERRED_CLASS: "3A",
  ALTERNATIVE_CLASSES: ["SL", "3E", "2A", "1A"],
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
// CONTROL STATE
// ==========================================
let scriptState = {
  running: false,
  paused: false,
  stopped: false,
  currentStage: "",
  startTime: null,
};

// ==========================================
// TIMING CONFIGURATION
// ==========================================
const TIMING = {
  // afterStation: () => 80 + Math.random() * 60,
  // afterAutocomplete: () => 150 + Math.random() * 100,
  // afterDate: () => 60 + Math.random() * 40,
  // afterDropdown: () => 80 + Math.random() * 40,
  // afterSearch: () => 1000 + Math.random() * 500,
  // afterClassSelect: () => 120 + Math.random() * 80,
  // beforeBook: () => 200 + Math.random() * 150,
  // afterBook: () => 700 + Math.random() * 300,
  // betweenFields: () => 50 + Math.random() * 40,
  // typing: () => 70 + Math.random() * 50,
  // shortWait: () => 30 + Math.random() * 20,
  // polling: () => 40 + Math.random() * 30,
  // beforeContinue: () => 500 + Math.random() * 300,
  afterStation: () => 1500 + Math.random() * 500, 
  afterAutocomplete: () => 500 + Math.random() * 200,
  afterDate: () => 1000 + Math.random() * 500,
  afterDropdown: () => 800 + Math.random() * 400,
  afterSearch: () => 2000 + Math.random() * 1000,
  afterClassSelect: () => 800 + Math.random() * 400,
  beforeBook: () => 1000 + Math.random() * 500,
  afterBook: () => 1500 + Math.random() * 500,
  betweenFields: () => 300 + Math.random() * 200,
  typing: () => 150 + Math.random() * 100, // Slower typing
  shortWait: () => 100 + Math.random() * 50,
  polling: () => 200 + Math.random() * 100,
  beforeContinue: () => 1000 + Math.random() * 500,
};

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const rWait = (fn) => wait(fn());

// ==========================================
// PAUSE/RESUME CONTROL
// ==========================================
async function checkPause() {
  while (scriptState.paused && !scriptState.stopped) {
    updateControlPanel();
    await wait(100);
  }
  if (scriptState.stopped) {
    throw new Error("Script stopped by user");
  }
}

function pauseScript() {
  if (!scriptState.running) {
    log("Script is not running", "warn");
    return;
  }
  scriptState.paused = true;
  log("⏸️  SCRIPT PAUSED - Press R to resume, S to stop", "warn");
  updateControlPanel();
}

function resumeScript() {
  if (!scriptState.paused) {
    log("Script is not paused", "warn");
    return;
  }
  scriptState.paused = false;
  log("▶️  SCRIPT RESUMED", "success");
  updateControlPanel();
}

function stopScript() {
  scriptState.stopped = true;
  scriptState.running = false;
  scriptState.paused = false;
  stopErrorMonitor();
  log("🛑 SCRIPT STOPPED", "error");
  updateControlPanel();
}

// ==========================================
// VISUAL CONTROL PANEL
// ==========================================
function createControlPanel() {
  const existing = document.getElementById("irctc-control-panel");
  if (existing) existing.remove();

  const panel = document.createElement("div");
  panel.id = "irctc-control-panel";
  panel.style.cssText = `
    position: fixed;
    top: 10px;
    right: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    z-index: 999999;
    font-family: 'Segoe UI', Arial, sans-serif;
    min-width: 280px;
    font-size: 13px;
  `;

  panel.innerHTML = `
    <div style="display: flex; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 10px;">
      <div style="font-size: 20px; margin-right: 8px;">🚂</div>
      <div>
        <div style="font-weight: bold; font-size: 14px;">IRCTC Tatkal Bot</div>
        <div id="status-text" style="font-size: 11px; opacity: 0.9;">Initializing...</div>
      </div>
    </div>
    
    <div id="stage-info" style="background: rgba(255,255,255,0.15); padding: 8px; border-radius: 5px; margin-bottom: 10px; font-size: 12px;">
      <div><strong>Stage:</strong> <span id="current-stage">Ready</span></div>
      <div><strong>Time:</strong> <span id="elapsed-time">0.0s</span></div>
    </div>
    
    <div style="display: flex; gap: 8px; margin-bottom: 8px;">
      <button id="start-btn" style="flex: 1; padding: 8px; background: #10b981; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 12px;">
        ▶️ START
      </button>
      <button id="pause-btn" style="flex: 1; padding: 8px; background: #f59e0b; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 12px;" disabled>
        ⏸️ PAUSE
      </button>
    </div>
    
    <div style="display: flex; gap: 8px;">
      <button id="resume-btn" style="flex: 1; padding: 8px; background: #3b82f6; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 12px;" disabled>
        ▶️ RESUME
      </button>
      <button id="stop-btn" style="flex: 1; padding: 8px; background: #ef4444; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 12px;" disabled>
        ⏹️ STOP
      </button>
    </div>
    
    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.3); font-size: 11px; opacity: 0.8;">
      <div>⌨️ Shortcuts:</div>
      <div>P = Pause | R = Resume | S = Stop</div>
    </div>
  `;

  document.body.appendChild(panel);

  document.getElementById("start-btn").onclick = () => {
    if (!scriptState.running) {
      runIRCTC();
    }
  };

  document.getElementById("pause-btn").onclick = pauseScript;
  document.getElementById("resume-btn").onclick = resumeScript;
  document.getElementById("stop-btn").onclick = stopScript;

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    if (e.key.toLowerCase() === "p") {
      pauseScript();
    } else if (e.key.toLowerCase() === "r") {
      resumeScript();
    } else if (e.key.toLowerCase() === "s") {
      stopScript();
    }
  });

  updateControlPanel();
}

function updateControlPanel() {
  const statusText = document.getElementById("status-text");
  const stageText = document.getElementById("current-stage");
  const timeText = document.getElementById("elapsed-time");
  const startBtn = document.getElementById("start-btn");
  const pauseBtn = document.getElementById("pause-btn");
  const resumeBtn = document.getElementById("resume-btn");
  const stopBtn = document.getElementById("stop-btn");

  if (!statusText) return;

  if (scriptState.stopped) {
    statusText.textContent = "🛑 Stopped";
    statusText.style.color = "#fca5a5";
  } else if (scriptState.paused) {
    statusText.textContent = "⏸️ Paused";
    statusText.style.color = "#fcd34d";
  } else if (scriptState.running) {
    statusText.textContent = "▶️ Running";
    statusText.style.color = "#86efac";
  } else {
    statusText.textContent = "⏹️ Ready";
    statusText.style.color = "#ffffff";
  }

  if (stageText) {
    stageText.textContent = scriptState.currentStage || "Ready";
  }

  if (timeText && scriptState.startTime) {
    const elapsed = ((Date.now() - scriptState.startTime) / 1000).toFixed(1);
    timeText.textContent = `${elapsed}s`;
  }

  if (startBtn) {
    startBtn.disabled = scriptState.running;
    startBtn.style.opacity = scriptState.running ? "0.5" : "1";
    startBtn.style.cursor = scriptState.running ? "not-allowed" : "pointer";
  }

  if (pauseBtn) {
    pauseBtn.disabled = !scriptState.running || scriptState.paused;
    pauseBtn.style.opacity =
      !scriptState.running || scriptState.paused ? "0.5" : "1";
    pauseBtn.style.cursor =
      !scriptState.running || scriptState.paused ? "not-allowed" : "pointer";
  }

  if (resumeBtn) {
    resumeBtn.disabled = !scriptState.paused;
    resumeBtn.style.opacity = !scriptState.paused ? "0.5" : "1";
    resumeBtn.style.cursor = !scriptState.paused ? "not-allowed" : "pointer";
  }

  if (stopBtn) {
    stopBtn.disabled = !scriptState.running;
    stopBtn.style.opacity = !scriptState.running ? "0.5" : "1";
    stopBtn.style.cursor = !scriptState.running ? "not-allowed" : "pointer";
  }
}

setInterval(() => {
  if (scriptState.running) {
    updateControlPanel();
  }
}, 100);

function setStage(stage) {
  scriptState.currentStage = stage;
  updateControlPanel();
  log(`STAGE: ${stage}`);
}

// ==========================================
// LOGGING
// ==========================================
function log(msg, type = "info") {
  const time = new Date().toISOString().substr(11, 8);
  const icon =
    type === "success"
      ? "✅"
      : type === "error"
      ? "❌"
      : type === "warn"
      ? "⚠️"
      : "ℹ️";
  console.log(`[${time}] ${icon} ${msg}`);
}

// ==========================================
// ELEMENT WAITING
// ==========================================
async function waitEl(selector, options = {}) {
  const {
    timeout = 5000,
    visible = true,
    enabled = true,
    description = selector,
  } = options;

  log(`Waiting for: ${description}`);
  const start = Date.now();

  while (Date.now() - start < timeout) {
    await checkPause();

    const el = document.querySelector(selector);

    if (el) {
      const isVisible = !visible || (el.offsetHeight > 0 && el.offsetWidth > 0);
      const isEnabled =
        !enabled || (!el.disabled && !el.hasAttribute("disabled"));
      const style = window.getComputedStyle(el);
      const notHidden =
        style.display !== "none" && style.visibility !== "hidden";

      if (isVisible && isEnabled && notHidden) {
        log(`Found: ${description}`, "success");
        return el;
      }
    }

    await wait(TIMING.polling());
  }

  throw new Error(`Timeout: ${description}`);
}

// ==========================================
// INPUT HELPERS
// ==========================================
function triggerEvents(el, value) {
  el.focus();
  if (value !== undefined) {
    el.value = value;
  }
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
  el.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true }));
  el.dispatchEvent(new Event("blur", { bubbles: true }));
}

async function typeText(el, text, description = "text") {
  log(`Typing: ${description}`);
  el.focus();
  el.value = "";

  for (let char of text) {
    await checkPause();
    el.value += char;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(
      new KeyboardEvent("keydown", { key: char, bubbles: true })
    );
    await rWait(TIMING.typing);
  }

  el.dispatchEvent(new Event("change", { bubbles: true }));
  el.blur();
  log(`Typed: ${description}`, "success");
}

function clickElement(el, description = "element") {
  log(`Clicking: ${description}`);

  const rect = el.getBoundingClientRect();
  const x = rect.left + rect.width * (0.4 + Math.random() * 0.2);
  const y = rect.top + rect.height * (0.4 + Math.random() * 0.2);

  el.dispatchEvent(
    new MouseEvent("mouseenter", { bubbles: true, clientX: x, clientY: y })
  );
  el.dispatchEvent(
    new MouseEvent("mouseover", { bubbles: true, clientX: x, clientY: y })
  );
  el.dispatchEvent(
    new MouseEvent("mousedown", {
      bubbles: true,
      clientX: x,
      clientY: y,
      button: 0,
    })
  );
  el.dispatchEvent(
    new MouseEvent("mouseup", {
      bubbles: true,
      clientX: x,
      clientY: y,
      button: 0,
    })
  );
  el.click();
  el.dispatchEvent(new MouseEvent("mouseleave", { bubbles: true }));

  log(`Clicked: ${description}`, "success");
}

// ==========================================
// ERROR DIALOG MONITOR
// ==========================================
let errorMonitor = null;

function startErrorMonitor() {
  if (errorMonitor) return;

  log("Starting error monitor");
  errorMonitor = setInterval(() => {
    const dialogs = document.querySelectorAll(
      '.ui-dialog, .p-dialog, [role="dialog"]'
    );

    for (let dialog of dialogs) {
      const text = dialog.textContent || "";
      if (text.match(/error|failed|process|invalid/i)) {
        log("Found error dialog, closing...", "warn");
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
    log("Stopped error monitor");
  }
}

// ==========================================
// PAGE DETECTION (Smart Resume)
// ==========================================
function detectCurrentPage() {
  // Check for passenger details page
  if (
    document.querySelector('input[placeholder="Name"]') &&
    document.querySelector('input[placeholder="Age"]')
  ) {
    return "passenger";
  }

  // Check for train list page
  if (document.querySelector("app-train-avl-enq")) {
    return "trainlist";
  }

  // Check for CAPTCHA page
  if (document.querySelector('img[alt*="Captcha"], img.captcha-img')) {
    return "captcha";
  }

  // Check for payment page
  if (document.querySelector('[name="paymentType"]')) {
    return "payment";
  }

  // Check for search form (home page)
  if (
    document.querySelector(
      'input[aria-label="Enter From station. Input is Mandatory."]'
    )
  ) {
    return "search";
  }

  return "unknown";
}

// ==========================================
// STATION FILLING
// ==========================================
async function fillStation(selector, code, fullName, description) {
  log(`Filling station: ${description}`);

  await checkPause();

  const input = await waitEl(selector, {
    timeout: 3000,
    description: `${description} input`,
  });

  input.value = "";
  input.focus();
  await rWait(TIMING.shortWait);

  for (let char of code) {
    await checkPause();
    input.value += char;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: char, bubbles: true })
    );
    await rWait(TIMING.typing);
  }

  input.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true }));

  log(`Waiting for autocomplete list...`);

  let autocompleteList = null;
  let attempts = 0;

  while (attempts < 3 && !autocompleteList) {
    await checkPause();
    await wait(200 + attempts * 100);

    autocompleteList = document.querySelector(".ui-autocomplete-items");

    if (!autocompleteList || autocompleteList.children.length === 0) {
      log(
        `Attempt ${attempts + 1}: Autocomplete not shown, retrying...`,
        "warn"
      );
      input.value = code + " ";
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true }));
      attempts++;
    } else {
      break;
    }
  }

  if (!autocompleteList || autocompleteList.children.length === 0) {
    throw new Error("Autocomplete list not appearing");
  }

  await rWait(TIMING.afterAutocomplete);

  const items = document.querySelectorAll(".ui-autocomplete-items li");
  log(`Found ${items.length} stations in autocomplete`);

  let clicked = false;

  for (let item of items) {
    const text = item.innerText.trim();
    if (text.includes(fullName) || text.includes(code)) {
      log(`Selecting: ${text.substring(0, 50)}...`);
      clickElement(item, `Station: ${code}`);
      clicked = true;
      break;
    }
  }

  if (!clicked && items.length > 0) {
    log(`Exact match not found, selecting first option`, "warn");
    clickElement(items[0], "First station");
    clicked = true;
  }

  if (!clicked) {
    throw new Error("No station items to click");
  }

  await wait(100);
  if (input.value.trim().length > 0) {
    log(`Station ${description} filled successfully`, "success");
  } else {
    log(`Warning: Station value may not be set`, "warn");
  }

  await rWait(TIMING.afterStation);
}

// ==========================================
// DATE FILLING
// ==========================================
async function fillDate(dateStr, description = "date") {
  log(`Filling date: ${dateStr}`);
  await checkPause();

  const [day, month, year] = dateStr.split("/").map(Number);
  const dateInput = await waitEl(".ui-calendar input", {
    timeout: 3000,
    description: "Date input",
  });

  clickElement(dateInput, "Date picker");
  await wait(200 + Math.random() * 150);

  await waitEl(".ui-datepicker", {
    timeout: 2000,
    description: "Date picker calendar",
  });

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
  const targetMonth = monthNames[month - 1];
  const targetYear = year.toString();

  let attempts = 0;
  while (attempts < 24) {
    await checkPause();
    const currMonthEl = document.querySelector(".ui-datepicker-month");
    const currYearEl = document.querySelector(".ui-datepicker-year");
    if (!currMonthEl || !currYearEl) break;

    const currMonth = currMonthEl.textContent.trim();
    const currYear = currYearEl.textContent.trim();

    if (currMonth === targetMonth && currYear === targetYear) {
      log(`Reached target month: ${targetMonth} ${targetYear}`, "success");
      break;
    }

    const curr = new Date(parseInt(currYear), monthNames.indexOf(currMonth), 1);
    const target = new Date(year, month - 1, 1);
    const btn =
      target > curr
        ? document.querySelector(".ui-datepicker-next")
        : document.querySelector(".ui-datepicker-prev");

    if (btn && !btn.classList.contains("ui-state-disabled")) {
      clickElement(btn, target > curr ? "Next month" : "Previous month");
      await wait(80 + Math.random() * 40);
    } else {
      break;
    }
    attempts++;
  }

  await wait(100 + Math.random() * 50);
  const dayLinks = document.querySelectorAll(
    ".ui-datepicker-calendar a.ui-state-default"
  );

  let dayClicked = false;
  for (let link of dayLinks) {
    if (parseInt(link.textContent.trim()) === day) {
      clickElement(link, `Day ${day}`);
      dayClicked = true;
      break;
    }
  }

  if (!dayClicked) throw new Error(`Day ${day} not found in calendar`);
  log(`Date filled: ${dateStr}`, "success");
  await rWait(TIMING.afterDate);
}

async function selectDropdown(selector, optionText, description) {
  log(`Selecting dropdown: ${description} = ${optionText}`);
  await checkPause();

  try {
    const dropdown = await waitEl(selector, {
      timeout: 2000,
      description: `Dropdown: ${description}`,
    });
    clickElement(dropdown, description);
    await wait(150 + Math.random() * 100);

    await waitEl(".ui-dropdown-items, p-dropdownitem", {
      timeout: 2000,
      description: "Dropdown options",
    });
    await wait(50 + Math.random() * 30);

    const options = document.querySelectorAll(
      ".ui-dropdown-items li, p-dropdownitem li"
    );
    for (let opt of options) {
      const text = opt.innerText.trim();
      if (text === optionText || text.includes(optionText)) {
        clickElement(opt, `Option: ${optionText}`);
        log(`Selected: ${optionText}`, "success");
        return true;
      }
    }
    log(`Option "${optionText}" not found`, "warn");
    return false;
  } catch (err) {
    log(`Dropdown error: ${err.message}`, "warn");
    return false;
  }
}

async function searchTrains() {
  log("Searching for trains...");
  const start = Date.now();
  const maxTime = 90000;

  while (Date.now() - start < maxTime) {
    await checkPause();
    try {
      const trainList = document.querySelector("app-train-avl-enq");
      if (trainList) {
        log("Already on train list page", "success");
        return true;
      }

      const searchBtn = document.querySelector("button.search_btn");
      if (searchBtn && !searchBtn.disabled) {
        clickElement(searchBtn, "Search button");
        await wait(300 + Math.random() * 200);

        const trains = document.querySelector("app-train-avl-enq");
        if (trains) {
          log("Trains loaded!", "success");
          return true;
        }
      }
      await wait(250 + Math.random() * 150);
    } catch (err) {
      log(`Search retry: ${err.message}`, "warn");
      await wait(500);
    }
  }
  throw new Error("Search failed after timeout");
}

async function findTrain(trainNumber) {
  log(`Looking for train: ${trainNumber}`);
  await checkPause();
  await rWait(TIMING.afterSearch);

  const trains = document.querySelectorAll("app-train-avl-enq");
  log(`Found ${trains.length} trains`);

  for (let train of trains) {
    const heading = train.querySelector(".train-heading strong");
    if (heading && heading.textContent.includes(trainNumber)) {
      log(`Found train ${trainNumber}!`, "success");
      return train;
    }
  }
  throw new Error(`Train ${trainNumber} not found`);
}

async function selectClass(trainEl, preferred, alternatives) {
  log(`Selecting class: ${preferred}`);
  await checkPause();

  const classMap = {
    SL: "Sleeper (SL)",
    "3E": "AC 3 Economy (3E)",
    "3A": "AC 3 Tier (3A)",
    "2A": "AC 2 Tier (2A)",
    "1A": "AC First Class (1A)",
  };

  const classesToTry = [preferred, ...alternatives];

  for (let code of classesToTry) {
    const name = classMap[code];
    if (!name) continue;

    log(`Trying class: ${name}`);

    const spans = trainEl.querySelectorAll("span.hidden-xs");
    for (let span of spans) {
      if (span.textContent.trim() === name) {
        clickElement(span, `Class: ${name}`);
        await rWait(TIMING.afterClassSelect);
        return code;
      }
    }

    const btns = trainEl.querySelectorAll(".pre-avl");
    for (let btn of btns) {
      const txt = btn.querySelector("strong")?.textContent;
      if (txt && txt.includes(name)) {
        clickElement(btn, `Class: ${name}`);
        await rWait(TIMING.afterClassSelect);
        return code;
      }
    }
  }
  throw new Error("No available class found");
}

async function handleDatePopup(targetDate) {
  log("Handling date popup...");
  await checkPause();
  await wait(100 + Math.random() * 50);

  const [day, month] = targetDate.split("/").map(Number);
  const monthShort = [
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
  ][month - 1];

  const dateOpts = Array.from(document.querySelectorAll(".pre-avl")).filter(
    (opt) => {
      const txt = opt.querySelector("strong")?.textContent;
      return txt && /\w+,\s+\d+\s+\w+/.test(txt);
    }
  );

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
    log("Target date not found, selecting first option", "warn");
    clickElement(dateOpts[0], "First date");
    await wait(400 + Math.random() * 200);
  }
}

async function bookTrain(trainEl) {
  log("Attempting to book train...");
  const start = Date.now();
  const maxTime = 90000;

  while (Date.now() - start < maxTime) {
    await checkPause();
    try {
      const passengerPage = document.querySelector('input[placeholder="Name"]');
      if (passengerPage) {
        log("Already on passenger page!", "success");
        return true;
      }

      let bookBtn = trainEl.querySelector("button.btnDefault.train_Search");
      if (!bookBtn) {
        const parent = trainEl.closest(".form-group");
        bookBtn = parent?.querySelector("button.btnDefault.train_Search");
      }
      if (!bookBtn) {
        const btns = trainEl.querySelectorAll("button");
        for (let b of btns) {
          if (
            b.textContent.includes("Book Now") ||
            b.textContent.includes("Book")
          ) {
            bookBtn = b;
            break;
          }
        }
      }

      if (bookBtn && !bookBtn.disabled) {
        bookBtn.scrollIntoView({ behavior: "smooth", block: "center" });
        await wait(100);
        clickElement(bookBtn, "Book Now");
        await wait(500 + Math.random() * 300);

        const loaded = document.querySelector('input[placeholder="Name"]');
        if (loaded) {
          log("Booking successful!", "success");
          return true;
        }
      }
      await wait(300 + Math.random() * 200);
    } catch (err) {
      log(`Book retry: ${err.message}`, "warn");
      await wait(500);
    }
  }
  throw new Error("Booking failed after timeout");
}

// ==========================================
// PASSENGER DETAILS
// ==========================================
async function fillPassengers(passengers) {
  log("Filling passenger details...");

  await checkPause();
  await rWait(TIMING.afterBook);

  for (let i = 0; i < passengers.length; i++) {
    await checkPause();

    const p = passengers[i];
    log(`Filling passenger ${i + 1}: ${p.name}`);

    const names = document.querySelectorAll('input[placeholder="Name"]');
    if (names[i]) {
      await typeText(names[i], p.name, `Passenger ${i + 1} name`);
      names[i].dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Escape",
          keyCode: 27,
          bubbles: true,
        })
      );
      await rWait(TIMING.betweenFields);
    }

    const ages = document.querySelectorAll('input[placeholder="Age"]');
    if (ages[i]) {
      triggerEvents(ages[i], p.age.toString());
      await rWait(TIMING.betweenFields);
    }

    const genders = document.querySelectorAll(
      'select[formcontrolname="passengerGender"]'
    );
    if (genders[i]) {
      genders[i].value = p.gender;
      genders[i].dispatchEvent(new Event("change", { bubbles: true }));
      await rWait(TIMING.betweenFields);
    }

    if (p.berthPreference) {
      const berths = document.querySelectorAll(
        'select[formcontrolname="passengerBerthChoice"]'
      );
      if (berths[i]) {
        berths[i].value = p.berthPreference;
        berths[i].dispatchEvent(new Event("change", { bubbles: true }));
        await rWait(TIMING.betweenFields);
      }
    }

    if (p.nationality) {
      const nats = document.querySelectorAll(
        'select[formcontrolname="passengerNationality"]'
      );
      if (nats[i]) {
        nats[i].value = p.nationality;
        nats[i].dispatchEvent(new Event("change", { bubbles: true }));
      }
    }
  }

  log("Passenger details filled", "success");
}

async function setPreferences(autoUpgrade, confirmBerths) {
  log("Setting booking preferences...");

  await checkPause();

  if (autoUpgrade) {
    const chk = document.querySelector('input[id="autoUpgradation"]');
    if (chk && !chk.checked) {
      clickElement(chk, "Auto-upgrade");
    }
  }

  if (confirmBerths) {
    const chk = document.querySelector('input[id="confirmberths"]');
    if (chk && !chk.checked) {
      clickElement(chk, "Confirm berths");
    }
  }

  await wait(100 + Math.random() * 50);
}

async function setPayment(type, upiId) {
  log("Setting payment method...");

  await checkPause();

  try {
    const loyalty = document.querySelector(
      'input[name="loyalityOperationType"][value="2"]'
    );
    if (loyalty?.checked) {
      const container = loyalty.closest("p-radiobutton");
      if (container) clickElement(container, "Disable loyalty");
    }
  } catch {}

  await wait(60 + Math.random() * 40);

  if (type === "UPI") {
    const upi = document.querySelector('input[name="paymentType"][value="2"]');
    if (upi && !upi.checked) {
      clickElement(upi, "UPI payment");
      await wait(80 + Math.random() * 40);

      const container = upi.closest("p-radiobutton");
      if (container && !upi.checked) {
        const box = container.querySelector(".ui-radiobutton-box");
        if (box) clickElement(box, "UPI radio box");
      }

      await wait(100 + Math.random() * 50);

      if (upiId) {
        const inp = document.querySelector(
          'input[placeholder*="UPI"], input[formcontrolname="vpaId"]'
        );
        if (inp) {
          triggerEvents(inp, upiId);
          log(`UPI ID set: ${upiId}`, "success");
        }
      }
    }
  }

  await wait(100 + Math.random() * 50);
}

// ==========================================
// CONTINUE BUTTON
// ==========================================
async function clickContinueButton() {
  log("Looking for Continue button...");

  await checkPause();

  log("Waiting for form validation...");
  await rWait(TIMING.beforeContinue);

  const selectors = [
    'button[type="submit"].train_Search.btnDefault',
    'button.train_Search.btnDefault[type="submit"]',
    "button.btnDefault.train_Search",
    'button[type="submit"]',
  ];

  let continueBtn = null;

  for (let selector of selectors) {
    continueBtn = document.querySelector(selector);
    if (continueBtn) {
      const btnText = continueBtn.textContent.trim().toLowerCase();
      if (btnText.includes("continue")) {
        log(`Found Continue button with selector: ${selector}`, "success");
        break;
      }
    }
    continueBtn = null;
  }

  if (!continueBtn) {
    log("Trying to find Continue button by text...", "warn");
    const allButtons = document.querySelectorAll("button");
    for (let btn of allButtons) {
      const text = btn.textContent.trim().toLowerCase();
      if (text.includes("continue") && btn.type === "submit") {
        continueBtn = btn;
        log("Found Continue button by text content", "success");
        break;
      }
    }
  }

  if (!continueBtn) {
    throw new Error("Continue button not found!");
  }

  if (continueBtn.disabled) {
    log("Continue button is disabled! Checking form validity...", "warn");

    const errors = document.querySelectorAll(
      ".error, .invalid, .ng-invalid.ng-dirty"
    );
    if (errors.length > 0) {
      log(`Found ${errors.length} form errors:`, "error");
      errors.forEach((err, i) => {
        if (err.offsetHeight > 0) {
          log(`  Error ${i + 1}: ${err.textContent.trim()}`, "error");
        }
      });
      throw new Error("Form has validation errors - Continue button disabled");
    }

    log("Waiting for button to enable...", "warn");
    let attempts = 0;
    while (continueBtn.disabled && attempts < 10) {
      await wait(500);
      attempts++;
    }

    if (continueBtn.disabled) {
      throw new Error("Continue button still disabled after waiting");
    }
  }

  log("Scrolling Continue button into view...");
  continueBtn.scrollIntoView({ behavior: "smooth", block: "center" });
  await wait(300 + Math.random() * 200);

  log("Clicking Continue button...");
  clickElement(continueBtn, "Continue Button");

  await wait(1000 + Math.random() * 500);

  const captchaImg = document.querySelector(
    'img[alt*="Captcha"], img.captcha-img'
  );
  if (captchaImg) {
    log("✅ CAPTCHA page reached!", "success");
    return "captcha";
  }

  const paymentOptions = document.querySelector('[name="paymentType"]');
  if (paymentOptions) {
    log("✅ Payment page reached!", "success");
    return "payment";
  }

  const errorDialog = document.querySelector(".ui-dialog, .p-dialog");
  if (errorDialog && errorDialog.textContent.includes("error")) {
    log("⚠️ Error dialog appeared after clicking Continue", "warn");
    return "error";
  }

  log("⚠️ Unclear which page we're on after clicking Continue", "warn");
  return "unknown";
}

// ==========================================
// MAIN EXECUTION (WITH SMART RESUME)
// ==========================================
async function runIRCTC() {
  if (scriptState.running) {
    log("Script is already running!", "warn");
    return;
  }

  scriptState.running = true;
  scriptState.paused = false;
  scriptState.stopped = false;
  scriptState.startTime = Date.now();
  updateControlPanel();

  try {
    log("========================================");
    log("🚀 IRCTC TATKAL SCRIPT STARTED");
    log("========================================");

    startErrorMonitor();

    // SMART DETECTION: Check which page we're on
    const currentPage = detectCurrentPage();
    log(`📍 Detected current page: ${currentPage.toUpperCase()}`, "success");

    let train = null;

    // Skip to appropriate stage based on current page
    if (currentPage === "passenger") {
      log("✨ Resuming from PASSENGER page", "success");
      log("========================================");

      // Jump directly to passenger filling
      setStage("Filling passenger details...");
      await fillPassengers(TICKET_CONFIG.PASSENGERS);

      await wait(120 + Math.random() * 80);
      await setPreferences(
        TICKET_CONFIG.AUTO_UPGRADATION,
        TICKET_CONFIG.CONFIRM_BERTHS_ONLY
      );

      await wait(100 + Math.random() * 50);
      await setPayment(TICKET_CONFIG.PAYMENT_TYPE, TICKET_CONFIG.UPI_ID);

      setStage("Clicking Continue button...");
      const nextPage = await clickContinueButton();

      const elapsed = ((Date.now() - scriptState.startTime) / 1000).toFixed(1);

      if (nextPage === "captcha") {
        setStage("✅ Ready for CAPTCHA!");
        log("========================================");
        log(`✅ REACHED CAPTCHA PAGE IN ${elapsed}s`, "success");
        log("🔐 Solve CAPTCHA manually and continue");
        log("========================================");
      } else if (nextPage === "payment") {
        setStage("✅ Payment page!");
        log("========================================");
        log(`✅ REACHED PAYMENT PAGE IN ${elapsed}s`, "success");
        log("💳 Complete payment manually");
        log("========================================");
      } else {
        setStage("⚠️ Check manually");
        log("========================================");
        log(`⚠️ FORM COMPLETED IN ${elapsed}s`, "warn");
        log("Please check the page and continue manually");
        log("========================================");
      }
    } else if (currentPage === "trainlist") {
      log("✨ Resuming from TRAIN LIST page", "success");
      log("========================================");

      // STAGE 3: TRAIN SELECTION
      setStage("Selecting train...");
      train = await findTrain(TICKET_CONFIG.TARGET_TRAIN_NUMBER);

      await selectClass(
        train,
        TICKET_CONFIG.PREFERRED_CLASS,
        TICKET_CONFIG.ALTERNATIVE_CLASSES
      );

      await handleDatePopup(TICKET_CONFIG.TRAVEL_DATE);
      await rWait(TIMING.beforeBook);

      // STAGE 4: BOOKING
      setStage("Booking train...");
      await bookTrain(train);

      // STAGE 5: PASSENGERS
      setStage("Filling passenger details...");
      await fillPassengers(TICKET_CONFIG.PASSENGERS);

      await wait(120 + Math.random() * 80);
      await setPreferences(
        TICKET_CONFIG.AUTO_UPGRADATION,
        TICKET_CONFIG.CONFIRM_BERTHS_ONLY
      );

      await wait(100 + Math.random() * 50);
      await setPayment(TICKET_CONFIG.PAYMENT_TYPE, TICKET_CONFIG.UPI_ID);

      // STAGE 6: CONTINUE BUTTON
      setStage("Clicking Continue button...");
      const nextPage = await clickContinueButton();

      const elapsed = ((Date.now() - scriptState.startTime) / 1000).toFixed(1);

      if (nextPage === "captcha") {
        setStage("✅ Ready for CAPTCHA!");
        log("========================================");
        log(`✅ REACHED CAPTCHA PAGE IN ${elapsed}s`, "success");
        log("🔐 Solve CAPTCHA manually and continue");
        log("========================================");
      } else if (nextPage === "payment") {
        setStage("✅ Payment page!");
        log("========================================");
        log(`✅ REACHED PAYMENT PAGE IN ${elapsed}s`, "success");
        log("💳 Complete payment manually");
        log("========================================");
      } else {
        setStage("⚠️ Check manually");
        log("========================================");
        log(`⚠️ FORM COMPLETED IN ${elapsed}s`, "warn");
        log("Please check the page and continue manually");
        log("========================================");
      }
    } else if (currentPage === "captcha") {
      log("✨ Already on CAPTCHA page!", "success");
      log("========================================");
      log("🔐 Solve CAPTCHA manually and continue", "success");
      log("========================================");
      setStage("✅ Ready for CAPTCHA!");
    } else if (currentPage === "payment") {
      log("✨ Already on PAYMENT page!", "success");
      log("========================================");
      log("💳 Complete payment manually", "success");
      log("========================================");
      setStage("✅ Payment page!");
    } else {
      // Start from beginning (search page)
      log("✨ Starting from SEARCH page", "success");
      log("========================================");

      // STAGE 1: FORM FILLING
      setStage("Filling search form...");
      await fillStation(
        'input[aria-label="Enter From station. Input is Mandatory."]',
        TICKET_CONFIG.FROM_STATION_CODE,
        TICKET_CONFIG.FROM_STATION_FULL,
        "FROM station"
      );

      await fillStation(
        'input[aria-label="Enter To station. Input is Mandatory."]',
        TICKET_CONFIG.TO_STATION_CODE,
        TICKET_CONFIG.TO_STATION_FULL,
        "TO station"
      );

      await fillDate(TICKET_CONFIG.TRAVEL_DATE, "Journey date");

      if (TICKET_CONFIG.TRAIN_CLASS !== "All Classes") {
        await selectDropdown(
          ".ng-tns-c76-10.ui-dropdown",
          TICKET_CONFIG.TRAIN_CLASS,
          "Class"
        );
      }

      if (TICKET_CONFIG.QUOTA !== "GENERAL") {
        await selectDropdown(
          ".ng-tns-c76-11.ui-dropdown",
          TICKET_CONFIG.QUOTA,
          "Quota"
        );
      }

      await rWait(TIMING.afterDropdown);

      // STAGE 2: SEARCH
      setStage("Searching trains...");
      await searchTrains();

      // STAGE 3: TRAIN SELECTION
      setStage("Selecting train...");
      train = await findTrain(TICKET_CONFIG.TARGET_TRAIN_NUMBER);

      await selectClass(
        train,
        TICKET_CONFIG.PREFERRED_CLASS,
        TICKET_CONFIG.ALTERNATIVE_CLASSES
      );

      await handleDatePopup(TICKET_CONFIG.TRAVEL_DATE);
      await rWait(TIMING.beforeBook);

      // STAGE 4: BOOKING
      setStage("Booking train...");
      await bookTrain(train);

      // STAGE 5: PASSENGERS
      setStage("Filling passenger details...");
      await fillPassengers(TICKET_CONFIG.PASSENGERS);

      await wait(120 + Math.random() * 80);
      await setPreferences(
        TICKET_CONFIG.AUTO_UPGRADATION,
        TICKET_CONFIG.CONFIRM_BERTHS_ONLY
      );

      await wait(100 + Math.random() * 50);
      await setPayment(TICKET_CONFIG.PAYMENT_TYPE, TICKET_CONFIG.UPI_ID);

      // STAGE 6: CONTINUE BUTTON
      setStage("Clicking Continue button...");
      const nextPage = await clickContinueButton();

      const elapsed = ((Date.now() - scriptState.startTime) / 1000).toFixed(1);

      if (nextPage === "captcha") {
        setStage("✅ Ready for CAPTCHA!");
        log("========================================");
        log(`✅ REACHED CAPTCHA PAGE IN ${elapsed}s`, "success");
        log("🔐 Solve CAPTCHA manually and continue");
        log("========================================");
      } else if (nextPage === "payment") {
        setStage("✅ Payment page!");
        log("========================================");
        log(`✅ REACHED PAYMENT PAGE IN ${elapsed}s`, "success");
        log("💳 Complete payment manually");
        log("========================================");
      } else {
        setStage("⚠️ Check manually");
        log("========================================");
        log(`⚠️ FORM COMPLETED IN ${elapsed}s`, "warn");
        log("Please check the page and continue manually");
        log("========================================");
      }
    }

    scriptState.running = false;
    stopErrorMonitor();
    updateControlPanel();
  } catch (err) {
    const elapsed = scriptState.startTime
      ? ((Date.now() - scriptState.startTime) / 1000).toFixed(1)
      : "0";

    if (err.message === "Script stopped by user") {
      setStage("🛑 Stopped by user");
      log("========================================");
      log("🛑 Script stopped by user", "warn");
      log("========================================");
    } else {
      setStage("❌ Error");
      log("========================================");
      log(`❌ ERROR after ${elapsed}s: ${err.message}`, "error");
      log("Stack trace:", "error");
      console.error(err);
      log("========================================");
    }

    scriptState.running = false;
    stopErrorMonitor();
    updateControlPanel();
  }
}

// ==========================================
// INITIALIZE
// ==========================================
createControlPanel();

log("========================================");
log("✨ IRCTC Tatkal Script Loaded!");
log("========================================");
log("📱 Control panel added to top-right");
log("🎮 Click START button when ready");
log("⌨️  Keyboard shortcuts:");
log("   P = Pause script");
log("   R = Resume script");
log("   S = Stop script");
log("========================================");
log("🔧 NEW: Smart Resume Feature!");
log("   ✓ Stop at any stage");
log("   ✓ Do manual selection");
log("   ✓ Click START - it resumes from current page");
log("========================================");
log("📍 Detects: Search, Train List, Passenger, CAPTCHA");
log("========================================");

// Export functions
window.irctcStart = runIRCTC;
window.irctcPause = pauseScript;
window.irctcResume = resumeScript;
window.irctcStop = stopScript;

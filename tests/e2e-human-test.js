/**
 * Human Simulation Test with Puppeteer
 * =====================================
 * Opens real Chromium browser, loads http://localhost:3000,
 * checks console logs, network errors, clicks presets,
 * submits form, and observes DOM updates and drought risk score.
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function runHumanTest() {
  console.log('--- Starting Human-Like Browser Automation Test ---');

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const page = await browser.newPage();

  // Track console logs and network failures
  const consoleMessages = [];
  const networkErrors = [];

  page.on('console', (msg) => {
    consoleMessages.push({ type: msg.type(), text: msg.text() });
    console.log(`[BROWSER CONSOLE ${msg.type().toUpperCase()}]:`, msg.text());
  });

  page.on('pageerror', (err) => {
    console.error('[BROWSER PAGE UNCAUGHT ERROR]:', err.message);
  });

  page.on('requestfailed', (req) => {
    networkErrors.push({ url: req.url(), error: req.failure().errorText });
    console.error(`[BROWSER NETWORK FAILED]: ${req.method()} ${req.url()} - ${req.failure().errorText}`);
  });

  page.on('response', (res) => {
    if (res.status() >= 400) {
      console.warn(`[BROWSER HTTP NON-200]: ${res.status()} ${res.url()}`);
    }
  });

  console.log('1. Navigating to http://localhost:3000 ...');
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle2' });

  // Check page title
  const title = await page.title();
  console.log('2. Page loaded! Title:', title);

  // Check connection status in UI
  await page.waitForSelector('#statusText');
  const statusText = await page.$eval('#statusText', (el) => el.textContent.trim());
  const statusDotClass = await page.$eval('#statusDot', (el) => el.className);
  console.log(`3. Connection Status in UI: "${statusText}" (class="${statusDotClass}")`);

  // Click on Borana preset: Yabelo
  console.log('4. User clicks "Yabelo" preset button...');
  await page.click('button[data-name="Yabelo"]');

  const latVal = await page.$eval('#inputLatitude', (el) => el.value);
  const lonVal = await page.$eval('#inputLongitude', (el) => el.value);
  const yrVal = await page.$eval('#inputYear', (el) => el.value);
  console.log(`   Form populated: lat=${latVal}, lon=${lonVal}, year=${yrVal}`);

  // User clicks "Check Drought Forecast" submit button
  console.log('5. User clicks "Check Drought Forecast" submit button...');
  await page.click('#btnSubmit');

  // Wait to see what happens
  console.log('6. Observing UI state transitions...');

  // Check if processing banner appeared
  const isProcessingVisible = await page.$eval('#processingCard', (el) => !el.classList.contains('hidden'));
  const processingText = await page.$eval('#processingMessage', (el) => el.textContent.trim());
  console.log(`   Processing card visible: ${isProcessingVisible}, message: "${processingText}"`);

  // Wait for either result section or error banner
  console.log('7. Waiting for prediction result or error banner...');
  try {
    await page.waitForFunction(
      () => {
        const resEl = document.getElementById('resultSection');
        const errEl = document.getElementById('generalError');
        const hasResult = resEl && !resEl.classList.contains('hidden');
        const hasError = errEl && !errEl.classList.contains('hidden') && errEl.textContent.trim().length > 0;
        return hasResult || hasError;
      },
      { timeout: 15000 }
    );
  } catch (timeoutErr) {
    console.error('TIMED OUT waiting for either resultSection or generalError!');
  }

  // Inspect what happened
  const resVisible = await page.$eval('#resultSection', (el) => !el.classList.contains('hidden'));
  const errVisible = await page.$eval('#generalError', (el) => !el.classList.contains('hidden'));
  const errText = await page.$eval('#generalError', (el) => el.textContent.trim());

  if (errVisible) {
    console.error(`ERROR BANNER DISPLAYED: "${errText}"`);
  }

  if (resVisible) {
    const badgeText = await page.$eval('#severityLabel', (el) => el.textContent.trim());
    const advisoryText = await page.$eval('#pumpAdvisory', (el) => el.textContent.trim());
    const probNormal = await page.$eval('#probNormalText', (el) => el.textContent.trim());
    const probMod = await page.$eval('#probModerateText', (el) => el.textContent.trim());
    const probSev = await page.$eval('#probSevereText', (el) => el.textContent.trim());
    const gridCell = await page.$eval('#metaGridCell', (el) => el.textContent.trim());
    const latency = await page.$eval('#metaLatency', (el) => el.textContent.trim());

    console.log('SUCCESS! Result displayed to human user:');
    console.log(`   Risk Badge:      ${badgeText}`);
    console.log(`   Advisory:        ${advisoryText}`);
    console.log(`   Probabilities:   Normal: ${probNormal}, Moderate: ${probMod}, Severe: ${probSev}`);
    console.log(`   Grid Cell:       ${gridCell}`);
    console.log(`   Engine Latency:  ${latency}`);
  }

  // Save screenshot for audit
  const screenshotPath = path.join(__dirname, '../human-test-screenshot.png');
  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log(`8. Saved audit screenshot to ${screenshotPath}`);

  await browser.close();
  console.log('--- Human Browser Automation Test Complete ---');

  if (errVisible) {
    process.exit(1);
  }
}

runHumanTest().catch((err) => {
  console.error('Fatal test error:', err);
  process.exit(1);
});

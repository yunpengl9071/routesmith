const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    executablePath: '/usr/bin/chromium-browser',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
  });
  
  const page = await browser.newPage();
  
  await page.goto(`file://${process.cwd()}/routesmith_for_browser.html`, {
    waitUntil: 'networkidle2'
  });
  
  // Wait for MathJax
  console.log('Waiting for MathJax...');
  try {
    await page.waitForFunction(() => {
      return typeof MathJax !== 'undefined' && document.querySelector('.MathJax');
    }, { timeout: 30000 });
    
    // Wait for typesetting
    await page.evaluate(() => MathJax.typesetPromise());
    await new Promise(resolve => setTimeout(resolve, 3000));
  } catch(e) {
    console.log('MathJax wait error:', e.message);
  }
  
  await page.pdf({
    path: 'puppeteer_rendered.pdf',
    format: 'A4',
    printBackground: true,
    margin: { top: '1in', bottom: '1in', left: '1in', right: '1in' }
  });
  
  console.log('PDF saved!');
  await browser.close();
})();

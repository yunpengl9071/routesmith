const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    executablePath: '/usr/bin/chromium-browser',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  await page.goto(`file://${process.cwd()}/proper_clean.html`, { waitUntil: 'networkidle2' });
  
  console.log('Waiting for MathJax...');
  await page.waitForFunction(() => typeof MathJax !== 'undefined', { timeout: 30000 });
  await page.evaluate(() => MathJax.typesetPromise());
  await new Promise(r => setTimeout(r, 5000));
  
  await page.pdf({
    path: 'proper_rendered.pdf',
    format: 'A4',
    printBackground: true,
    margin: { top: '1in', bottom: '1in' }
  });
  
  console.log('Done!');
  await browser.close();
})();

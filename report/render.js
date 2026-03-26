const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  
  // Load the HTML file
  await page.goto(`file://${process.cwd()}/routesmith_for_browser.html`, {
    waitUntil: 'networkidle0'
  });
  
  // Wait for MathJax to finish rendering
  console.log('Waiting for MathJax...');
  await page.waitForFunction(() => {
    return window.MathJax && window.MathJax.typesetPromise && window.MathJax.typesetPromise.length === 0;
  }, { timeout: 60000 });
  
  // Additional wait for rendering
  await new Promise(resolve => setTimeout(resolve, 5000));
  
  // Generate PDF
  await page.pdf({
    path: 'mathjax_rendered.pdf',
    format: 'A4',
    printBackground: true,
    margin: { top: '1in', bottom: '1in', left: '1in', right: '1in' }
  });
  
  console.log('PDF saved!');
  await browser.close();
})();

// Browser integration checks for the local, privacy-preserving site guide.
import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright-core');
const here = path.dirname(fileURLToPath(import.meta.url));
const site = path.resolve(here, '../../docs/site');
const out = path.join(here, 'out/guide-bot');
fs.mkdirSync(out, { recursive: true });
const mime = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.json': 'application/json' };
const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const relative = url.pathname.endsWith('/') ? url.pathname + 'index.html' : url.pathname;
  const file = path.resolve(site, '.' + relative);
  if (!file.startsWith(site + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({
  ...(process.env.CHROME_EXECUTABLE ? { executablePath: process.env.CHROME_EXECUTABLE } : { channel: 'chrome' }),
  args: ['--disable-dev-shm-usage'],
});

async function insideViewport(locator, width, height) {
  const r = await locator.boundingBox();
  assert(r && r.x >= 0 && r.y >= 0 && r.x + r.width <= width + 1 && r.y + r.height <= height + 1,
    `Outside ${width}x${height}: ${JSON.stringify(r)}`);
}

async function assertFloatingGuide(page, width, height) {
  const position = await page.evaluate(() => {
    const bot = document.querySelector('.guide-bot');
    const r = document.querySelector('.guide-bot__launcher').getBoundingClientRect();
    return { docked: bot.classList.contains('is-docked'), dock: !!document.querySelector('.guide-bot__dock'),
      width: r.width, height: r.height, top: r.top,
      headerBottom: document.querySelector('.site-header').getBoundingClientRect().bottom,
      horizontalOverflow: document.documentElement.scrollWidth > innerWidth };
  });
  assert(!position.docked && !position.dock, `Guide must float at ${width}x${height}`);
  assert.equal(position.width, 52);
  assert.equal(position.height, 52);
  assert(position.top >= position.headerBottom, 'Default launcher must stay below the header');
  await insideViewport(page.locator('.guide-bot__launcher'), width, height);
  assert.equal(position.horizontalOverflow, false);
}

try {
  for (const [width, height] of [[1440, 900], [1440, 450], [1366, 768], [1280, 720], [1024, 768],
    [900, 600], [768, 540], [390, 844], [320, 480], [640, 280]]) {
    const page = await browser.newPage({ viewport: { width, height }, reducedMotion: 'reduce' });
    const foreign = [], errors = [];
    page.on('request', request => {
      const url = new URL(request.url());
      if (url.origin !== origin && !['data:', 'blob:', 'about:'].includes(url.protocol)) foreign.push(request.url());
    });
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*.mp4', route => route.abort());
    await page.goto(origin + '/', { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => document.fonts.ready);
    await assertFloatingGuide(page, width, height);
    await page.locator('#integrated').scrollIntoViewIfNeeded();
    await page.waitForTimeout(50);
    await assertFloatingGuide(page, width, height);
    await page.screenshot({ path: path.join(out, `floating-${width}x${height}.png`) });
    const launcher = page.locator('.guide-bot__launcher');
    const panel = page.locator('.guide-bot__panel');
    if (width <= 1023) {
      await page.locator('#menu-btn').click();
      assert.equal(await page.locator('#menu-btn').getAttribute('aria-expanded'), 'true');
    }
    await launcher.focus();
    await page.keyboard.press('Enter');
    assert.equal(await launcher.getAttribute('aria-expanded'), 'true');
    assert.equal(await page.locator('#menu-btn').getAttribute('aria-expanded'), 'false');
    await insideViewport(panel, width, height);
    await insideViewport(page.locator('.guide-bot__form input'), width, height);
    if (width <= 680) {
      assert.equal(await panel.getAttribute('aria-modal'), 'true');
      assert.equal(await page.locator('.guide-bot__backdrop').isVisible(), true);
      await page.locator('.guide-bot__form button').focus();
      await page.keyboard.press('Tab');
      assert.equal(await page.locator('.guide-bot__language').evaluate(el => document.activeElement === el), true);
      await page.keyboard.press('Shift+Tab');
      assert.equal(await page.locator('.guide-bot__form button').evaluate(el => document.activeElement === el), true);
    }
    if (await page.locator('[data-guide-question="methods"]').isVisible()) {
      await page.locator('[data-guide-question="methods"]').click();
    } else {
      await page.locator('#guide-bot-input').fill('methods');
      await page.locator('.guide-bot__form').evaluate(form => form.requestSubmit());
    }
    await page.waitForFunction(() => document.querySelector('.guide-bot__message:last-child').textContent.includes('DTI is the default'));
    await page.locator('.guide-bot__language').click();
    assert.match(await page.locator('.guide-bot__message').last().innerText(), /微结构默认使用 DTI/);
    await page.locator('#guide-bot-input').fill('如何准备输入数据？');
    await page.locator('.guide-bot__form').evaluate(form => form.requestSubmit());
    await page.waitForFunction(() => document.querySelector('.guide-bot__message:last-child').textContent.includes('DICOM'));
    assert.match(await page.locator('.guide-bot__message').last().innerText(), /DICOM/);
    await page.locator('#guide-bot-input').fill('如何使用 Apple MPS 或显卡？');
    await page.locator('.guide-bot__form').evaluate(form => form.requestSubmit());
    await page.waitForFunction(() => document.querySelector('.guide-bot__message:last-child').textContent.includes('Compute device'));
    assert.equal(await page.locator('.guide-bot__message').last().locator('a').getAttribute('href'), '#start');
    await page.locator('#guide-bot-input').fill('<img src=x onerror=window.injected=1>');
    await page.locator('.guide-bot__form').evaluate(form => form.requestSubmit());
    await page.waitForFunction(() => document.querySelector('.guide-bot__message:last-child').textContent.includes('没有可靠的预设回答'));
    assert.equal(await page.evaluate(() => window.injected), undefined);
    assert.equal(await page.locator('.guide-bot__messages img').count(), 0);
    assert.match(await page.locator('.guide-bot__message').last().innerText(), /没有可靠的预设回答/);
    await page.keyboard.press('Escape');
    assert.equal(await panel.isHidden(), true);
    assert.equal(await launcher.getAttribute('aria-expanded'), 'false');
    assert.equal(await launcher.evaluate(el => document.activeElement === el), true);
    if (width <= 680) {
      await launcher.click();
      await page.locator('.guide-bot__backdrop').click({ position: { x: 2, y: 2 } });
      assert.equal(await panel.isHidden(), true);
      assert.equal(await page.evaluate(() => document.body.classList.contains('guide-bot-open')), false);
      assert.equal(await launcher.evaluate(el => document.activeElement === el), true);
    } else {
      await launcher.click();
      await page.mouse.click(width - 8, height - 8);
      assert(await panel.isHidden(), 'Clicking outside the desktop guide should return to reading');
    }
    assert.deepEqual(foreign, []);
    assert.deepEqual(errors, []);
    console.log(`PASS guide interactions and viewport ${width}x${height}`);
    await page.close();
  }
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  await page.route('**/*.mp4', route => route.abort());
  await page.goto(origin + '/', { waitUntil: 'domcontentloaded' });
  const launcher = page.locator('.guide-bot__launcher');
  const box = await launcher.boundingBox();
  await page.mouse.move(box.x + 26, box.y + 26);
  await page.mouse.down();
  await page.mouse.move(1310, 700, { steps: 8 });
  await page.mouse.up();
  assert(await page.locator('.guide-bot__panel').isHidden(), 'drag must not open the guide');
  const after = await launcher.boundingBox();
  assert(after.x > box.x + 100, 'launcher did not follow the drag');
  await page.setViewportSize({ width: 390, height: 844 });
  await insideViewport(launcher, 390, 844);
  assert(await page.locator('.guide-bot').evaluate(el => !el.classList.contains('is-docked')));
  const mobileBox = await launcher.boundingBox();
  await page.mouse.move(mobileBox.x + 26, mobileBox.y + 26);
  await page.mouse.down();
  await page.mouse.move(110, 600, { steps: 8 });
  await page.mouse.up();
  const mobileAfter = await launcher.boundingBox();
  assert(Math.abs(mobileAfter.x - mobileBox.x) > 20, 'The guide must remain draggable on small screens');
  await page.locator('#start').scrollIntoViewIfNeeded();
  await page.waitForTimeout(50);
  const afterScroll = await launcher.boundingBox();
  assert(Math.abs(afterScroll.x - mobileAfter.x) < 1 && Math.abs(afterScroll.y - mobileAfter.y) < 1,
    'Scrolling must not move the floating guide into the header');
  await page.setViewportSize({ width: 1440, height: 900 });
  assert(await page.locator('.guide-bot').evaluate(el => !el.classList.contains('is-docked')));
  const restored = await launcher.boundingBox();
  assert(Math.abs(restored.x - mobileAfter.x) < 1 && Math.abs(restored.y - mobileAfter.y) < 1,
    'Preserve the latest user-selected floating position after resizing');
  await page.setViewportSize({ width: 390, height: 844 });
  await launcher.click();
  await insideViewport(page.locator('.guide-bot__panel'), 390, 844);
  if (await page.locator('[data-guide-question="references"]').isVisible()) {
    await page.locator('[data-guide-question="references"]').click();
  } else {
    await page.locator('#guide-bot-input').fill('references');
    await page.locator('.guide-bot__form').evaluate(form => form.requestSubmit());
  }
  await page.locator('.guide-bot__message a').last().click();
  assert.equal(new URL(page.url()).hash, '#integrated');
  assert(await page.locator('.guide-bot__panel').isHidden());
  await page.close();
  const desktop = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  await desktop.route('**/*.mp4', route => route.abort());
  await desktop.goto(origin + '/', { waitUntil: 'domcontentloaded' });
  await desktop.locator('.guide-bot__launcher').click();
  const header = await desktop.locator('.guide-bot__header').boundingBox();
  const beforeHeaderDrag = await desktop.locator('.guide-bot__launcher').boundingBox();
  await desktop.mouse.move(header.x + 80, header.y + 10);
  await desktop.mouse.down();
  await desktop.mouse.move(header.x + 260, header.y + 70, { steps: 8 });
  await desktop.mouse.up();
  const afterHeaderDrag = await desktop.locator('.guide-bot__launcher').boundingBox();
  assert(afterHeaderDrag.x > beforeHeaderDrag.x + 100, 'dragging the header must move the mascot');
  assert(await desktop.locator('.guide-bot__panel').isVisible());
  await desktop.close();

  const animated = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: 'no-preference' });
  await animated.route('**/*.mp4', route => route.abort());
  await animated.goto(origin + '/', { waitUntil: 'domcontentloaded' });
  const motion = await animated.evaluate(() => {
    const bot = document.querySelector('.guide-bot');
    const launcher = document.querySelector('.guide-bot__launcher');
    const eye = getComputedStyle(document.querySelector('.guide-bot__face'), '::before');
    const pulse = launcher.getAnimations().find(a => a.animationName === 'guide-pulse');
    pulse.pause();
    pulse.currentTime = 0;
    return { float: getComputedStyle(bot).animationDuration,
      pulse: getComputedStyle(launcher).animationDuration, shadow: getComputedStyle(launcher).boxShadow,
      border: getComputedStyle(launcher).borderTopColor, background: getComputedStyle(launcher).backgroundColor,
      eyes: [eye.width, eye.height, eye.animationDuration] };
  });
  assert.equal(motion.float, '6s');
  assert.equal(motion.pulse, '2.8s');
  assert.match(motion.shadow, /rgba\(78, 168, 255, 0\.35\).*20px/);
  assert.equal(motion.border, 'rgb(245, 247, 251)');
  assert.equal(motion.background, 'rgb(11, 11, 15)');
  assert.deepEqual(motion.eyes, ['6px', '16px', '5s']);
  await animated.mouse.move(1200, 320);
  await animated.waitForTimeout(300);
  const mid = Number.parseFloat(await animated.locator('.guide-bot__launcher').evaluate(el => el.style.getPropertyValue('--eye-x')));
  await animated.waitForTimeout(1000);
  const end = Number.parseFloat(await animated.locator('.guide-bot__launcher').evaluate(el => el.style.getPropertyValue('--eye-x')));
  assert(mid > 0 && mid < 5 && end > mid && end <= 5 && end > 4.4, `eye smoothing: ${mid}, ${end}`);
  const source = fs.readFileSync(path.join(site, 'assets/js/guide-bot.js'), 'utf8');
  assert.match(source, /\}, 150\);/);
  assert.match(source, /eyeTarget\.x - eyeCurrent\.x\) \* \.05/);
  await animated.close();

  const festive = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await festive.addInitScript(() => {
    const OriginalDate = Date;
    window.Date = class extends OriginalDate {
      constructor(...args) { super(...(args.length ? args : ['2026-12-24T12:00:00'])); }
    };
  });
  await festive.route('**/*.mp4', route => route.abort());
  await festive.goto(origin + '/', { waitUntil: 'domcontentloaded' });
  assert.equal(await festive.locator('.guide-bot').evaluate(el => el.classList.contains('is-festive')), true);
  assert.match(await festive.locator('.guide-bot__hat').evaluate(el => getComputedStyle(el).backgroundImage), /guide-bot-hat\.svg/);
  const hat = fs.readFileSync(path.join(site, 'assets/img/guide-bot-hat.svg'), 'utf8').trimEnd();
  assert.equal(createHash('sha256').update(hat).digest('hex'), 'bde7919badcfc32e4ce97185d0fd30adee2adb52c64d6736ec12ab16e7712da4');
  await festive.close();
  const noJs = await browser.newPage({ javaScriptEnabled: false });
  await noJs.route('**/*.mp4', route => route.abort());
  await noJs.goto(origin + '/', { waitUntil: 'domcontentloaded' });
  assert(await noJs.locator('.guide-bot__launcher').isHidden());
  assert(await noJs.locator('h1').isVisible());
  await noJs.close();
  console.log('PASS floating guide, mobile drag, resize, navigation, focus trap, motion, festive asset, and no-JavaScript fallback');
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}

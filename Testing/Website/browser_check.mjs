// Headless Chrome checks for the SlicerDSpace project site (SPEC section 7.2).
//
// Run from the repository root:
//   npm ci --prefix Testing/Website
//   node Testing/Website/browser_check.mjs
//
// Serves docs/site/ with Range support (like GitHub Pages),
// opens it in Google Chrome (channel 'chrome': Playwright's Chromium lacks H.264/AAC)
// and writes screenshots to Testing/Website/out/ (git-ignored).

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const here = path.dirname(fileURLToPath(import.meta.url));
const DOCS = path.resolve(here, '..', '..', 'docs', 'site');
const OUT = path.resolve(here, 'out');
const PERF_COUNT = JSON.parse(fs.readFileSync(path.join(DOCS, 'assets', 'data', 'performance.json'), 'utf8')).records.length;
fs.mkdirSync(OUT, { recursive: true });

const coreDir = process.env.PLAYWRIGHT_CORE;
const require = createRequire(import.meta.url);
const { chromium } = coreDir ? require(path.resolve(coreDir)) : require('playwright-core');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mp4': 'video/mp4',
  '.srt': 'application/x-subrip',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
};

// ---------------------------------------------------------------- server
function serve() {
  const server = http.createServer((req, res) => {
    let urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (urlPath.startsWith('/SlicerDSpace/')) urlPath = urlPath.slice('/SlicerDSpace'.length);
    if (urlPath.endsWith('/')) urlPath += 'index.html';
    const file = path.resolve(DOCS, '.' + urlPath);
    if (!file.startsWith(DOCS + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      const nf = path.join(DOCS, '404.html');
      res.writeHead(404, { 'Content-Type': MIME['.html'] });
      res.end(fs.readFileSync(nf));
      return;
    }
    const size = fs.statSync(file).size;
    const type = MIME[path.extname(file)] || 'application/octet-stream';
    const range = req.headers.range;
    const headers = { 'Content-Type': type, 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-store' };
    if (range) {
      const m = /^bytes=(\d*)-(\d*)$/.exec(range);
      if (!m) { res.writeHead(416, { 'Content-Range': `bytes */${size}` }); res.end(); return; }
      let start = m[1] === '' ? size - Number(m[2]) : Number(m[1]);
      let end = m[1] === '' || m[2] === '' ? size - 1 : Math.min(Number(m[2]), size - 1);
      if (start > end || start >= size) { res.writeHead(416, { 'Content-Range': `bytes */${size}` }); res.end(); return; }
      res.writeHead(206, { ...headers, 'Content-Range': `bytes ${start}-${end}/${size}`, 'Content-Length': end - start + 1 });
      if (req.method === 'HEAD') { res.end(); return; }
      fs.createReadStream(file, { start, end }).on('error', () => res.destroy()).pipe(res);
      return;
    }
    res.writeHead(200, { ...headers, 'Content-Length': size });
    if (req.method === 'HEAD') { res.end(); return; }
    fs.createReadStream(file).on('error', () => res.destroy()).pipe(res);
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

// ---------------------------------------------------------------- reporting
const results = [];
function record(ok, name, detail = '') {
  results.push({ ok, name, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${ok ? '' : ' - ' + detail}`);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function scrollThrough(page) {
  const height = await page.evaluate(() => document.documentElement.scrollHeight);
  const vh = await page.evaluate(() => window.innerHeight);
  for (let y = 0; y < height; y += Math.floor(vh * 0.7)) {
    await page.evaluate((top) => window.scrollTo(0, top), y);
    await sleep(120);
  }
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
  await sleep(300);
  await page.evaluate(() => window.scrollTo(0, 0));
  await sleep(200);
}

// ---------------------------------------------------------------- one viewport
async function runViewport(browser, origin, name, width, height, opts) {
  const tag = `${name}`;
  const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const consoleErrors = [];
  const failed = [];
  const foreign = [];
  const mediaRequests = [];
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', (e) => consoleErrors.push('pageerror: ' + e.message));
  page.on('request', (r) => {
    const u = new URL(r.url());
    if (/\/media\/[^/?#]+\.mp4$/.test(u.pathname)) mediaRequests.push(r.url());
    if (u.origin !== origin && u.protocol !== 'data:' && u.protocol !== 'blob:' && u.protocol !== 'about:') foreign.push(r.url());
  });
  page.on('requestfailed', (r) => {
    const err = r.failure()?.errorText || '';
    // Chrome cancels and re-issues byte-range requests for video. Nothing else may fail.
    if (err.includes('ERR_ABORTED') && /\/media\/[^/?#]+\.mp4$/.test(new URL(r.url()).pathname)) return;
    failed.push(`${r.url()} (${err})`);
  });
  page.on('response', (r) => {
    if (r.status() >= 400) failed.push(`${r.url()} (HTTP ${r.status()})`);
  });

  await page.goto(origin + '/index.html', { waitUntil: 'load' });
  await page.waitForTimeout(400);
  record(mediaRequests.length === 0, `${tag}: no video bytes requested before playback`, String(mediaRequests.length));

  // 2. no horizontal scroll
  const w = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth }));
  record(w.sw <= w.iw, `${tag}: no horizontal scroll (scrollWidth ${w.sw} <= innerWidth ${w.iw})`,
    `scrollWidth ${w.sw} > innerWidth ${w.iw}`);

  if (opts.overflowOnly) {
    await scrollThrough(page);
    const w2 = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth }));
    record(w2.sw <= w2.iw, `${tag}: no horizontal scroll after scrolling through`, `scrollWidth ${w2.sw} > ${w2.iw}`);
    const wide = await page.evaluate(() => {
      const iw = window.innerWidth;
      return [...document.querySelectorAll('body *')].filter((el) => {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) return false;
        if (el.closest('pre') || el.closest('.sr-only') || el.closest('.table-wrap') || el.closest('.perf-table-wrap')) return false;
        if (el.closest('.results-scroll') && !el.matches('.results-scroll')) return false;
        return r.right > iw + 0.5 || r.left < -0.5;
      }).slice(0, 5).map((el) => el.tagName.toLowerCase() + '.' + el.className);
    });
    record(wide.length === 0, `${tag}: no element sticks out of the viewport`, wide.join(', '));
    await page.screenshot({ path: path.join(OUT, `site-${width}.png`), fullPage: true });
    record(consoleErrors.length === 0 && failed.length === 0 && foreign.length === 0,
      `${tag}: no console error, failed request or foreign host`,
      JSON.stringify({ consoleErrors, failed, foreign }));
    await context.close();
    return;
  }

  // 3. images
  await scrollThrough(page);
  const demoBoxesStartClosed = await page.locator('details.demo-box').evaluateAll(items =>
    items.length === 2 && items.every(d => !d.open && getComputedStyle(d.querySelector('.demo-box-body')).display === 'none'));
  record(demoBoxesStartClosed, `${tag}: demonstration Box 2 and Box 3 modules start collapsed by default`);
  await page.locator('details.demo-box').evaluateAll(items => items.forEach(d => { d.open = true; }));
  for (const image of await page.locator('img').all()) {
    if (await image.evaluate(i => !i.complete || i.naturalWidth === 0)) await image.scrollIntoViewIfNeeded();
  }
  await page.waitForFunction(() => [...document.images].every((i) => i.complete && i.naturalWidth > 0), null, { timeout: 15000 }).catch(() => {});
  const imgs = await page.evaluate(() => [...document.images].map((i) => ({
    src: i.getAttribute('src'), ok: i.complete && i.naturalWidth > 0,
  })));
  await page.locator('details.demo-box').evaluateAll(items => items.forEach(d => { d.open = false; }));
  const bad = imgs.filter((i) => !i.ok).map((i) => i.src);
  record(bad.length === 0, `${tag}: all ${imgs.length} images loaded`, 'not loaded: ' + bad.join(', '));
  const webpSources = await page.locator('picture > img').evaluateAll(items => items.map(image => image.currentSrc));
  record(webpSources.length === 1 && webpSources.every(source => source.endsWith('.webp')),
    `${tag}: lossless WebP alternatives are selected for PNG figures`, JSON.stringify(webpSources));
  record((await page.locator('#choose img').getAttribute('src')).split('?')[0] === 'assets/img/workflow-overview.svg',
    `${tag}: route overview uses the paper workflow as a vector image`);

  const wordmark = await page.locator('.product-wordmark').evaluate((img) => {
    const box = img.getBoundingClientRect();
    return { src: img.getAttribute('src'), width: box.width, height: box.height,
      nativeRatio: img.naturalWidth / img.naturalHeight, ratio: box.width / box.height,
      right: box.right, viewport: innerWidth };
  });
  record(wordmark.src === 'assets/img/slicerdspace-wordmark-light.svg' && wordmark.width > 0
      && wordmark.height > 0 && wordmark.right <= wordmark.viewport
      && Math.abs(wordmark.ratio / wordmark.nativeRatio - 1) < .02,
    `${tag}: vector title loads and scales without distortion`, JSON.stringify(wordmark));

  const frames = await page.evaluate(() => [...document.querySelectorAll('.tile, .tile img, .tile .chip, .shot, .shot img, .fig a, .film-frame')].map(el => {
    const style = getComputedStyle(el);
    return { name: el.className || el.tagName, border: [style.borderTopWidth, style.borderRightWidth,
      style.borderBottomWidth, style.borderLeftWidth], radii: [style.borderTopLeftRadius,
      style.borderTopRightRadius, style.borderBottomLeftRadius, style.borderBottomRightRadius] };
  }));
  record(frames.every(item => [...item.border, ...item.radii].every(value => parseFloat(value) === 0)),
    `${tag}: scientific images, video and result plates have no rounded frames`, JSON.stringify(frames));
  const separators = await page.locator('.tile').evaluateAll(items => items.every(el => {
    const style = getComputedStyle(el, '::after');
    return style.backgroundImage.includes('data:image/svg+xml') && style.pointerEvents === 'none'
      && getComputedStyle(el).backgroundColor === 'rgba(0, 0, 0, 0)';
  }));
  record(separators, `${tag}: open result plates use noninteractive curved separators`);

  const branding = await page.locator('.foot-brand img').evaluateAll(imgs => imgs.map(img => {
    const r = img.getBoundingClientRect();
    return { src: img.getAttribute('src'), width: r.width, height: r.height,
      ratio: (r.width / r.height) / (img.naturalWidth / img.naturalHeight) };
  }));
  record(branding.length === 1 && branding.every(img =>
    img.src === 'assets/img/slicerdspace-wordmark-light.svg' && img.width >= 180 && img.height > 0
      && Math.abs(img.ratio - 1) < .02),
    `${tag}: footer uses an undistorted wordmark`, JSON.stringify(branding));
  record(await page.locator('.site-header img, .site-header .brand, .chip-status').count() === 0,
    `${tag}: header has no logo or pre-release badge`);
  const center = await page.evaluate(() => {
    const mid = el => { const r = el.getBoundingClientRect(); return (r.left + r.right) / 2; };
    return { title: mid(document.querySelector('.product-wordmark')) - innerWidth / 2,
      actions: mid(document.querySelector('#top .btn-row')) - innerWidth / 2,
      navigation: innerWidth < 1024 ? 0 : mid(document.querySelector('.site-nav ul')) - innerWidth / 2 };
  });
  record(Object.values(center).every(offset => Math.abs(offset) < 1),
    `${tag}: wordmark, actions and desktop navigation are centered`, JSON.stringify(center));
  const heroContents = await page.locator('#top .container').evaluate(el => [...el.children].map(child => child.tagName));
  record(JSON.stringify(heroContents) === JSON.stringify(['H1', 'DIV']) &&
      await page.locator('#top .btn-row a').count() === 2 &&
      await page.locator('#top + #film video').count() === 1,
    `${tag}: wordmark, two actions and film have no intervening intro copy`, JSON.stringify(heroContents));
  record(await page.locator('.guide-bot__launcher').isVisible() &&
      await page.locator('.guide-bot__dock, .guide-bot.is-docked').count() === 0,
    `${tag}: guide stays floating without header docking`);

  const overview = await page.locator('.results-overview img').evaluate((img) => {
    const box = img.getBoundingClientRect();
    return { src: img.getAttribute('src'), width: img.naturalWidth, height: img.naturalHeight,
      fit: getComputedStyle(img).objectFit, ratio: box.width / box.height };
  });
  record(overview.src === 'assets/img/results-overview.svg?v=ukf-coverage-2' && overview.width > 0
      && Math.abs(overview.ratio - 1830 / 2036) < .01,
    `${tag}: complete vector results overview is uncropped`, JSON.stringify(overview));
  record(await page.locator('#outputs article.tile').count() === 0,
    `${tag}: results are one scientific plate, not separate gallery cards`);
  const plateScroll = await page.locator('.results-scroll').evaluate(el => {
    const r = el.getBoundingClientRect();
    return { left: r.left, right: r.right, viewport: innerWidth,
      content: el.scrollWidth, visible: el.clientWidth };
  });
  record(plateScroll.left >= 0 && plateScroll.right <= plateScroll.viewport &&
      (width > 700 || plateScroll.content > plateScroll.visible),
    `${tag}: small screens pan only the results plate, not the page`, JSON.stringify(plateScroll));
  await page.locator('.results-overview').screenshot({ path: path.join(OUT, `results-${width}.png`),
    style: '.site-header { visibility: hidden !important; }' });

  const w2 = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth }));
  record(w2.sw <= w2.iw, `${tag}: no horizontal scroll after scrolling through`, `scrollWidth ${w2.sw} > ${w2.iw}`);

  // Header fits and the right nav mode is on.
  const hdr = await page.evaluate(() => {
    const inner = document.querySelector('.header-inner');
    const btn = document.getElementById('menu-btn');
    const nav = document.getElementById('site-nav');
    const vis = (el) => { const s = getComputedStyle(el); return s.display !== 'none' && s.visibility !== 'hidden'; };
    return { fits: inner.scrollWidth <= inner.clientWidth, menuVisible: vis(btn), navVisible: vis(nav) };
  });
  const narrow = width <= 1023;
  record(hdr.fits, `${tag}: header content fits`, 'header-inner overflows');
  record(hdr.menuVisible === narrow, `${tag}: menu button ${narrow ? 'shown' : 'hidden'}`, JSON.stringify(hdr));
  if (!narrow) record(hdr.navVisible, `${tag}: nav links visible`, JSON.stringify(hdr));

  // 4. video
  const vid = await page.evaluate(() => {
    const v = document.getElementById('film-video');
    return {
      readyState: v.readyState, duration: v.duration, videoWidth: v.videoWidth,
      can: v.canPlayType('video/mp4; codecs="avc1.640032, mp4a.40.2"'),
    };
  });
  record(mediaRequests.length === 0 && vid.readyState === 0,
    `${tag}: scrolling through the page does not fetch film metadata or video`, JSON.stringify(vid));
  record(vid.can !== '', `${tag}: canPlayType for avc1.640032 + mp4a.40.2 is "${vid.can}"`, 'empty');

  // The play button replaces the native controls at start.
  const pre = await page.evaluate(() => {
    const v = document.getElementById('film-video');
    const b = document.querySelector('.film-play');
    return { controls: v.hasAttribute('controls'), btn: !!b && !b.hidden, label: b ? b.textContent.trim() : '' };
  });
  record(!pre.controls && pre.btn && pre.label === 'Play film',
    `${tag}: play button shown, controls removed (label "${pre.label}")`, JSON.stringify(pre));

  // 6. menu with the keyboard (narrow only)
  if (narrow) {
    await page.evaluate(() => window.scrollTo(0, 0));
    const btn = page.locator('#menu-btn');
    await btn.focus();
    await page.keyboard.press('Enter');
    await page.waitForTimeout(150);
    let s = await page.evaluate(() => ({
      exp: document.getElementById('menu-btn').getAttribute('aria-expanded'),
      shown: getComputedStyle(document.getElementById('site-nav')).display !== 'none',
    }));
    record(s.exp === 'true' && s.shown, `${tag}: Enter opens the menu`, JSON.stringify(s));
    await page.screenshot({ path: path.join(OUT, `menu-open-${width}.png`) });
    await page.keyboard.press('Escape');
    await page.waitForTimeout(150);
    s = await page.evaluate(() => ({
      exp: document.getElementById('menu-btn').getAttribute('aria-expanded'),
      shown: getComputedStyle(document.getElementById('site-nav')).display !== 'none',
      focusOnBtn: document.activeElement && document.activeElement.id === 'menu-btn',
    }));
    record(s.exp === 'false' && !s.shown && s.focusOnBtn, `${tag}: Escape closes the menu and returns focus`, JSON.stringify(s));
    await page.keyboard.press('Space');
    await page.waitForTimeout(150);
    s = await page.evaluate(() => document.getElementById('menu-btn').getAttribute('aria-expanded'));
    await page.keyboard.press('Space');
    await page.waitForTimeout(150);
    const s2 = await page.evaluate(() => document.getElementById('menu-btn').getAttribute('aria-expanded'));
    record(s === 'true' && s2 === 'false', `${tag}: Space opens and closes the menu`, `${s} then ${s2}`);
    // A link click closes the menu and moves to the section.
    await page.keyboard.press('Enter');
    await page.waitForTimeout(150);
    await page.locator('#site-nav a[href="#integrated"]').click();
    // Smooth scrolling over a long page takes a while: wait until the position is stable.
    let last = -1;
    for (let i = 0; i < 40; i++) {
      await page.waitForTimeout(150);
      const y = await page.evaluate(() => Math.round(window.scrollY));
      if (y === last) break;
      last = y;
    }
    s = await page.evaluate(() => ({
      exp: document.getElementById('menu-btn').getAttribute('aria-expanded'),
      top: Math.round(document.getElementById('integrated').getBoundingClientRect().top),
    }));
    record(s.exp === 'false' && s.top >= 0 && s.top < 200, `${tag}: link click closes the menu and lands on #integrated (top ${s.top})`, JSON.stringify(s));
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(300);
  }

  // 7. screenshots (before playback, page fully revealed)
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(OUT, `site-${width}.png`), fullPage: true });
  await page.screenshot({ path: path.join(OUT, `fold-${width}.png`) });

  // 5. play button starts playback
  await page.locator('#film').scrollIntoViewIfNeeded();
  await page.evaluate(() => { document.getElementById('film-video').muted = true; });
  await page.locator('.film-play').click();
  const t0 = await page.evaluate(() => document.getElementById('film-video').currentTime);
  let advanced = false;
  let t1 = t0;
  for (let i = 0; i < 30 && !advanced; i++) {
    await sleep(100);
    t1 = await page.evaluate(() => document.getElementById('film-video').currentTime);
    advanced = t1 > t0 + 0.2;
  }
  const post = await page.evaluate(() => {
    const v = document.getElementById('film-video');
    const b = document.querySelector('.film-play');
    return { controls: v.hasAttribute('controls'), btnHidden: b.hidden, paused: v.paused };
  });
  record(advanced && !post.paused, `${tag}: click on the play button starts playback (currentTime ${t0.toFixed(2)} -> ${t1.toFixed(2)})`, JSON.stringify({ t0, t1, ...post }));
  record(post.controls && post.btnHidden, `${tag}: controls shown and play button hidden after start`, JSON.stringify(post));
  const played = await page.locator('#film-video').evaluate(v => ({ duration: v.duration, width: v.videoWidth }));
  record(Math.abs(played.duration - 64.8) < 0.05 && played.width === 1920,
    `${tag}: playback retains the complete 1080p film`, JSON.stringify(played));
  await page.screenshot({ path: path.join(OUT, `playing-${width}.png`) });

  // Ended state: jump near the end and wait for the button to return.
  await page.evaluate(() => { const v = document.getElementById('film-video'); v.currentTime = Math.max(0, v.duration - 0.6); });
  await page.waitForFunction(() => !document.querySelector('.film-play').hidden, null, { timeout: 8000 }).catch(() => {});
  const ended = await page.evaluate(() => {
    const b = document.querySelector('.film-play');
    return { hidden: b.hidden, label: b.textContent.trim(), controls: document.getElementById('film-video').hasAttribute('controls') };
  });
  record(!ended.hidden && ended.label === 'Play again' && !ended.controls, `${tag}: on ended the button returns as "${ended.label}"`, JSON.stringify(ended));

  // Hero button: scrolls to the film and plays.
  await page.evaluate(() => { const v = document.getElementById('film-video'); v.currentTime = 0; v.pause(); window.scrollTo(0, 0); });
  await page.waitForTimeout(300);
  await page.locator('[data-play-film]').click();
  await page.waitForTimeout(1500);
  const hero = await page.evaluate(() => {
    const v = document.getElementById('film-video');
    return { paused: v.paused, t: v.currentTime, top: Math.round(document.getElementById('film').getBoundingClientRect().top) };
  });
  record(!hero.paused && hero.t > 0.2 && hero.top >= -5 && hero.top < 250, `${tag}: "Watch the film" scrolls to #film and plays (top ${hero.top}, t ${hero.t.toFixed(2)})`, JSON.stringify(hero));

  // The description comes first, followed by source code and the video download.
  await page.evaluate(() => { document.getElementById('film-video').pause(); });
  const minimalFilm = await page.evaluate(() => !document.getElementById('transcript') &&
    document.querySelectorAll('#film .meta-row a').length === 2 &&
    document.querySelector('#film .meta-row').firstElementChild.matches('span') &&
    document.querySelector('#film .meta-row').children[1].matches('a.source-link') &&
    document.querySelector('#film .source-link').href === 'https://github.com/NeuroDMRI/SlicerDSpace' &&
    document.querySelector('#film .source-link').textContent.trim() === 'Source code' &&
    !!document.querySelector('#film .source-link svg[aria-hidden="true"]') &&
    document.querySelector('#film .meta-row span').textContent === 'Diffusion MRI workflows in 3D Slicer.' &&
    document.querySelectorAll('#film a[href$=".srt"]').length === 0);
  record(minimalFilm, `${tag}: workflow description precedes source code and video download`);

  record(consoleErrors.length === 0 && failed.length === 0 && foreign.length === 0,
    `${tag}: no console error, failed request or foreign host`,
    JSON.stringify({ consoleErrors, failed, foreign }));

  await context.close();
}

// ---------------------------------------------------------------- 404 and no-JS
async function run404AndNoJs(browser, origin) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, javaScriptEnabled: false });
  const page = await context.newPage();
  const failed = [];
  page.on('requestfailed', (r) => failed.push(r.url()));
  await page.goto(origin + '/index.html', { waitUntil: 'load' });
  const nojs = await page.evaluate(() => {
    const v = document.getElementById('film-video');
    const hidden = [...document.querySelectorAll('.reveal')].filter((e) => getComputedStyle(e).opacity !== '1').length;
    return { controls: v.hasAttribute('controls'), hidden, copyShown: [...document.querySelectorAll('.copy')].some((b) => getComputedStyle(b).display !== 'none') };
  }).catch((e) => ({ error: String(e) }));
  record(nojs.controls === true && nojs.hidden === 0 && nojs.copyShown === false,
    'no-JS: native controls stay, all sections visible, no dead Copy buttons', JSON.stringify(nojs));
  await page.screenshot({ path: path.join(OUT, 'nojs-1440.png') });
  await context.close();

  const c2 = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const p2 = await c2.newPage();
  const errs = [];
  p2.on('console', (m) => { if (m.type() === 'error' && !/404/.test(m.text())) errs.push(m.text()); });
  const resp = await p2.goto(origin + '/404.html', { waitUntil: 'load' });
  await p2.waitForTimeout(300);
  const sw = await p2.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
  record(sw && errs.length === 0, '404 page renders without overflow or console errors', JSON.stringify({ sw, errs }));
  await p2.screenshot({ path: path.join(OUT, '404-390.png') });
  await c2.close();
}

// Method details expand directly underneath their measurement row.
async function runPerformance(browser, origin, width, noJs = false) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, javaScriptEnabled: !noJs });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('requestfailed', (request) => {
    // Chrome cancels and re-issues byte-range requests for video. Nothing else may fail.
    if ((request.failure()?.errorText || '').includes('ERR_ABORTED') && /\/media\/[^/?#]+\.mp4$/.test(new URL(request.url()).pathname)) return;
    errors.push(request.url());
  });
  page.on('request', (request) => { const u = new URL(request.url()); if (u.origin !== origin && u.protocol !== 'data:' && u.protocol !== 'blob:' && u.protocol !== 'about:') errors.push('External request: ' + request.url()); });
  page.on('response', (response) => { if (response.status() >= 400) errors.push('HTTP ' + response.status() + ': ' + response.url()); });
  await page.goto(origin + '/SlicerDSpace/index.html', { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
  const prefix = `performance ${width}px${noJs ? ' no JS' : ''}`;
  const evidence = page.locator('#performance-evidence');
  await evidence.scrollIntoViewIfNeeded();
  record(await evidence.evaluate(el => el.tagName === 'DIV' && !el.closest('details')) &&
    await page.locator('#performance-evidence .perf-table').isVisible(), prefix + ': complete comparison table is visible by default');
  record(await page.locator('.perf-table .perf-record').count() === PERF_COUNT &&
    await page.locator('.perf-table .perf-record').evaluateAll(rows => rows.every(row => getComputedStyle(row).display !== 'none')),
    prefix + ': all timing rows are visible without expanding anything');
  record(await page.locator('.perf-chart, .perf-filters, #perf-count').count() === 0,
    prefix + ': no filter controls, count or highlighted bar charts');
  record(await page.locator('.perf-methods').count() === 0, prefix + ': no separate method-details section');
  record(await page.locator('.perf-toggle[aria-expanded="false"]').count() === PERF_COUNT &&
    await page.locator('.perf-detail-row.is-open').count() === 0 &&
    await page.locator('.perf-detail-row').evaluateAll(rows => rows.every(row => row.hidden && getComputedStyle(row).display === 'none')),
    prefix + ': all method and verification details start hidden, including without JavaScript');
  record(await page.locator('.perf-toggle-icon').evaluateAll(icons => icons.every(icon => icon.textContent === '+')),
    prefix + ': each timing row has an explicit plus control');
  if (noJs) {
    record(await page.locator('.perf-toggle:disabled').count() === PERF_COUNT,
      prefix + ': failed or disabled scripting does not expose details or leave active broken buttons');
  }
  record(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), prefix + ': no page overflow with the table visible');
  if (!noJs) {
    await page.locator('.perf-toggle[aria-controls="method-freewater"] .perf-toggle-icon').click();
    record(await page.locator('#method-freewater').isVisible() &&
      await page.locator('#method-freewater').evaluate(el => el.previousElementSibling.dataset.record === 'freewater'),
      prefix + ': measurement expands inside the table below its own row');
    const bodyWidth = await page.locator('#method-freewater .perf-method-body').evaluate(el => el.getBoundingClientRect().width);
    record(bodyWidth <= width - 32, prefix + ': expanded description fits the viewport', String(bodyWidth));
    record(await page.locator('.perf-toggle[aria-controls="method-freewater"] .perf-toggle-icon').innerText() === '-',
      prefix + ': opening a record changes its plus to minus');
    await page.locator('.perf-toggle[aria-controls="method-freewater"]').focus();
    await page.keyboard.press('Enter');
    record(await page.locator('#method-freewater').isHidden(), prefix + ': Enter collapses inline details');
    await page.keyboard.press('Space');
    record(await page.locator('#method-freewater').isVisible(), prefix + ': Space reopens inline details');
    await page.locator('.perf-toggle[aria-controls="method-wma"]').click();
    record(await page.locator('#method-wma').isVisible() && await page.locator('#method-freewater').isVisible(),
      prefix + ': independent records can be compared inline');
    await page.locator('.perf-toggle[aria-controls="method-wma"]').click();
    record(await page.locator('#method-wma').isHidden(), prefix + ': clicking again closes that record');
    await page.goto(origin + '/SlicerDSpace/index.html#method-dki', { waitUntil: 'load' });
    await page.reload({ waitUntil: 'load' });
    record(await page.locator('.perf-detail-row').evaluateAll(rows => rows.every(row => row.hidden)) &&
      await page.locator('.perf-toggle[aria-controls="method-dki"]').getAttribute('aria-expanded') === 'false',
      prefix + ': a fresh load never expands details from a retained URL fragment');
    const response = await context.request.get(origin + '/SlicerDSpace/assets/data/performance.json');
    const data = await response.json();
    record(response.ok() && data.records.length === PERF_COUNT && !JSON.stringify(data).includes('/data/chengjin/'), prefix + ': sanitized evidence download');
    // Both the direct fragment and footer navigate to the already-visible table.
    await page.goto(origin + '/SlicerDSpace/index.html#performance', { waitUntil: 'load' });
    record(await page.locator('#performance-evidence .perf-table').isVisible(), prefix + ': #performance shows the complete table');
    await page.locator('footer a[href="#performance"]').click();
    record(await page.locator('#performance-evidence .perf-table').isVisible(), prefix + ': footer link reaches the visible table');
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: path.join(OUT, `performance-${width}${noJs ? '-nojs' : ''}.png`), fullPage: true });
  record(errors.length === 0, prefix + ': no errors or foreign requests', errors.join('; '));
  await context.close();
}

// A failed stylesheet, script, or root-class initializer must never expose details.
async function runPerformanceFailures(browser, origin) {
  for (const file of ['index.html', 'zh.html']) {
    for (const mode of ['no scripts or styles', 'failed performance script', 'no root JS class']) {
      const context = await browser.newContext({ viewport: { width: 390, height: 900 },
        javaScriptEnabled: mode !== 'no scripts or styles', reducedMotion: 'reduce' });
      const page = await context.newPage();
      if (mode === 'no scripts or styles') {
        await page.route('**/*.css*', route => route.abort());
      } else if (mode === 'failed performance script') {
        await page.route('**/performance.js*', route => route.abort());
      }
      await page.goto(origin + '/' + file + '#performance', { waitUntil: 'load' });
      if (mode === 'no root JS class') {
        await page.evaluate(() => document.documentElement.classList.remove('js'));
      }
      record(await page.locator('.perf-record').count() === PERF_COUNT &&
        await page.locator('.perf-table').isVisible() &&
        await page.locator('.perf-detail-row').evaluateAll((rows, expected) => rows.length === expected &&
          rows.every(row => row.hidden && getComputedStyle(row).display === 'none'), PERF_COUNT),
        `${file} ${mode}: timing table remains visible and every detail stays hidden`);
      if (mode === 'no root JS class') {
        const toggle = page.locator('.perf-toggle[aria-controls="method-freewater"]');
        await toggle.locator('.perf-toggle-icon').click();
        record(await page.locator('#method-freewater').isVisible() &&
          await page.locator('.perf-detail-row[hidden]').count() === PERF_COUNT - 1,
          `${file} ${mode}: explicit plus opens only the selected detail`);
        await toggle.click();
        record(await page.locator('.perf-detail-row[hidden]').count() === PERF_COUNT,
          `${file} ${mode}: minus restores the compact timing-only table`);
      } else {
        record(await page.locator('.perf-toggle:disabled').count() === PERF_COUNT,
          `${file} ${mode}: unavailable toggles are safely disabled`);
      }
      await context.close();
    }
  }
}

// Each method links directly to its primary paper or official documentation.
async function runReferenceLinks(browser, origin) {
  const refs = JSON.parse(fs.readFileSync(path.resolve(DOCS, '../website/references.json'), 'utf8')).references;
  const referenceById = new Map(refs.map(item => [String(item.id), item]));
  for (const [width, noJs] of [[1440, false], [390, false], [390, true]]) {
    const context = await browser.newContext({ viewport: { width, height: 900 }, javaScriptEnabled: !noJs });
    const page = await context.newPage();
    await page.goto(origin + '/index.html#integrated', { waitUntil: 'load' });
    const links = await page.locator('.int-table .method-reference').evaluateAll(items => items.map(a => ({
      id: a.dataset.reference, href: a.href, text: a.textContent.trim(), title: a.title,
      target: a.target, rel: a.rel,
    })));
    record(links.length > 35 && links.every(link => {
      const source = referenceById.get(link.id);
      return source && link.href === source.url && link.title === source.citation
        && link.target === '_blank' && link.rel.includes('noopener')
        && link.text === `[${link.id}]`;
    }), `references ${width}px${noJs ? ' no JS' : ''}: all method links match reviewed primary sources`);
    record(await page.locator('#cite, .ref-list, a[href^="#ref-"]').count() === 0 &&
      await page.locator('.int-table [data-reference="21"]').getAttribute('href') === referenceById.get('21').url,
      `references ${width}px${noJs ? ' no JS' : ''}: direct DOI links replace the standalone bibliography`);
    record(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      `references ${width}px${noJs ? ' no JS' : ''}: linked methods table does not widen the page`);
    const footer = page.locator('.site-footer');
    record(await footer.locator('.foot-nav').count() === 1 &&
      !/About the film|Rosetta|pre-release/.test(await footer.innerText()),
      `footer ${width}px${noJs ? ' no JS' : ''}: compact public-facing links and no production notes`);
    await footer.screenshot({ path: path.join(OUT, `footer-${width}${noJs ? '-nojs' : ''}.png`) });
    await context.close();
  }
}

async function runFooter(browser, origin) {
  for (const file of ['index.html', 'zh.html']) {
    for (const [width, noJs] of [[1440, false], [1024, false], [768, false], [640, false], [390, false], [320, false], [390, true]]) {
      const context = await browser.newContext({ viewport: { width, height: 900 }, javaScriptEnabled: !noJs, reducedMotion: 'reduce' });
      const page = await context.newPage();
      await page.goto(origin + '/' + file, { waitUntil: 'load' });
      const footer = page.locator('.site-footer');
      await footer.scrollIntoViewIfNeeded();
      await page.evaluate(() => document.fonts.ready);
      await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
      const prefix = `footer ${file} ${width}px${noJs ? ' no JS' : ''}`;
      const layout = await footer.evaluate(el => {
        const line = getComputedStyle(el, '::before');
        const headerLine = getComputedStyle(document.querySelector('.site-header'), '::after');
        const links = [...el.querySelectorAll('a')].map(a => a.getBoundingClientRect());
        const bounds = el.getBoundingClientRect();
        const overlaps = links.some((a, i) => links.slice(i + 1).some(b =>
          a.left < b.right - .5 && b.left < a.right - .5 &&
          a.top < b.bottom - .5 && b.top < a.bottom - .5));
        return {
          rainbow: line.content !== 'none' && line.backgroundImage === headerLine.backgroundImage &&
            line.backgroundImage.includes('linear-gradient') && line.height === '2px' && line.opacity === headerLine.opacity,
          fits: document.documentElement.scrollWidth <= innerWidth && links.every(a =>
            a.left >= 0 && a.right <= innerWidth + .5 && a.top >= bounds.top && a.bottom <= bounds.bottom + .5),
          overlaps,
        };
      });
      record(layout.rainbow, prefix + ': full-width rainbow line matches the header');
      record(layout.fits && !layout.overlaps, prefix + ': wordmark, navigation and legal links fit without overlap');
      if (width <= 680 && !noJs) {
        const clearOfGuide = await footer.evaluate(el => {
          const launcher = document.querySelector('.guide-bot__launcher').getBoundingClientRect();
          return [...el.querySelectorAll('a')].every(a => {
            const r = a.getBoundingClientRect();
            return r.right <= launcher.left || r.left >= launcher.right ||
              r.bottom <= launcher.top || r.top >= launcher.bottom;
          });
        });
        record(clearOfGuide, prefix + ': floating guide leaves footer links unobstructed at the page end');
      }
      record(await footer.locator('.foot-nav ul').count() === 2 &&
        await footer.locator('.foot-links a').count() === 2 &&
        await footer.locator('.foot-top-link').count() === 1,
        prefix + ': grouped navigation, legal links and back-to-top are distinct');
      await footer.screenshot({ path: path.join(OUT, `footer-redesign-${file}-${width}${noJs ? '-nojs' : ''}.png`) });
      await footer.locator('.foot-top-link').click();
      record(new URL(page.url()).hash === '#top' && await page.locator('.product-wordmark').isVisible(),
        prefix + ': back-to-top link reaches the hero');
      await context.close();
    }
  }
}

// Run times and the optimization table stay on the home page.
async function runHomePerformanceNavigation(browser, origin) {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const page = await context.newPage();
  await page.goto(origin + '/SlicerDSpace/index.html#runtime', { waitUntil: 'load' });
  const images = page.locator('#runtime img');
  record(await images.count() === 1 && await page.locator('img[src*="fig-runtime-optimizations"]').count() === 0,
    'home page: only the stage-time figure remains, without the optimization image');
  await page.locator('footer a[href="#performance"]').click();
  record(new URL(page.url()).pathname === '/SlicerDSpace/index.html' &&
    new URL(page.url()).hash === '#performance' && await page.locator('.perf-table').isVisible(),
    'home page: evidence link reaches the visible table on the same page', page.url());
  const retired = await context.request.get(origin + '/SlicerDSpace/performance.html');
  record(retired.status() === 404, 'retired performance page is not published', String(retired.status()));
  await context.close();
}

async function runSkills(browser, origin) {
  for (const width of [1440, 390, 320]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } });
    const page = await context.newPage();
    await page.goto(origin + '/index.html#skills', { waitUntil: 'load' });
    const skills = page.locator('#skills');
    await skills.scrollIntoViewIfNeeded();
    await page.waitForFunction(() => document.getElementById('start').classList.contains('in'));
    record(await skills.locator('.start-route-text').innerText() === 'Install, run and review with your Agent.' &&
      await skills.locator('.start-recommended').innerText() === '(Recommended)',
      `skills ${width}px: visible install-and-use introduction`);
    const headings = await page.locator('#start .start-route-title').evaluateAll(elements => {
      const styles = elements.map(el => getComputedStyle(el));
      return elements.length === 3 && elements.every(el => el.tagName === 'H3') &&
        elements.map(el => el.querySelector('.start-route-marker').textContent).join('') === 'ABC' &&
        styles.every(style => style.fontSize === styles[0].fontSize &&
          style.fontWeight === styles[0].fontWeight && style.lineHeight === styles[0].lineHeight);
    });
    record(headings, `skills ${width}px: A/B/C routes have matching heading typography and hierarchy`);
    const accents = await page.locator('#start .start-route-marker, #start .start-recommended')
      .evaluateAll(elements => elements.length === 4 && elements.every(el => getComputedStyle(el).color === 'rgb(169, 191, 208)'));
    record(accents, `skills ${width}px: route markers and recommendation share the muted blue-grey accent`);
    const timeline = await skills.locator('.agent-steps').evaluate(el => {
      const steps = [...el.children];
      const first = getComputedStyle(steps[0], '::before');
      const manual = getComputedStyle(document.querySelector('#start > .container > .start-steps .start-step'), '::before');
      return steps.length === 3 && steps.every(step => step.classList.contains('start-step')) &&
        first.width === manual.width && first.borderRadius === manual.borderRadius &&
        getComputedStyle(steps[0], '::after').width === '1px' &&
        getComputedStyle(steps[2], '::after').display === 'none';
    });
    record(timeline, `skills ${width}px: three numbered steps reuse the manual setup's rail and markers`);
    record(await skills.locator('.doc-links a').count() === 3 &&
      await page.locator('#start #skills').count() === 1 &&
      await page.locator('#site-nav a[href="#start"]').count() === 1 &&
      await page.locator('#site-nav a[href="#skills"]').count() === 0,
      `skills ${width}px: prompts and reference links are part of Get started`);
    const fit = await page.evaluate(() => {
      return [...document.querySelectorAll('#skills .code pre')].every(
        (el) => el.scrollWidth <= el.clientWidth) && document.documentElement.scrollWidth <= innerWidth;
    });
    record(fit, `skills ${width}px: prompt wraps without horizontal overflow`);
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    record(await skills.locator('[data-code]').count() === 2,
      `skills ${width}px: installation and analysis have separate prompts`);
    record(await page.locator('.skill-capabilities, .skills-grid').count() === 0 &&
      await page.locator('#cli').evaluate(el => !el.closest('details')),
      `skills ${width}px: no capability panels; CLI instructions are always expanded`);
    for (const id of ['agent-install-prompt', 'agent-analysis-prompt']) {
      const copyBtn = skills.locator('[data-code]').filter({ has: page.locator('#' + id) }).locator('.copy');
      await copyBtn.click();
      await page.waitForTimeout(80);
      const copied = (await page.evaluate(() => navigator.clipboard.readText())).replace(/\r\n/g, '\n');
      const expected = (await page.locator('#' + id).textContent()).replace(/\r\n/g, '\n');
      record(copied === expected &&
        copied.includes('https://github.com/NeuroDMRI/SlicerDSpace/blob/main/.agents/skills/slicerdspace/SKILL.md'),
        `skills ${width}px: ${id} copies intact with its Skill URL`);
    }
    await skills.screenshot({ path: path.join(OUT, `skills-${width}.png`) });
    await context.close();
  }
}

async function runDemonstrations(browser, origin) {
  for (const [file, width] of [['index.html', 1440], ['index.html', 390], ['zh.html', 1440], ['zh.html', 390]]) {
    const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce' });
    const page = await context.newPage();
    await page.route('**/*.mp4', route => route.abort());
    await page.goto(`${origin}/${file}#demonstrations`, { waitUntil: 'load' });
    const section = page.locator('#demonstrations');
    await section.scrollIntoViewIfNeeded();
    const box2 = page.locator('#demo-box-2');
    const box3 = page.locator('#demo-box-3');
    const prefix = `demonstrations ${file} ${width}px`;
    record(await box2.evaluate(d => !d.open) && await box3.evaluate(d => !d.open) &&
      await box2.locator('.demo-box-body').isHidden() && await box3.locator('.demo-box-body').isHidden(),
      `${prefix}: Box 2 and Box 3 start collapsed by default`);
    await box2.locator('.demo-box-summary').click();
    record(await box2.evaluate(d => d.open) && await box2.locator('#demo-box2-app1').isVisible() &&
      await box2.locator('#demo-box2-app2').isVisible() && await box3.evaluate(d => !d.open),
      `${prefix}: clicking Box 2 expands only Box 2`);
    await box3.locator('.demo-box-summary').click();
    record(await box3.evaluate(d => d.open) && await box3.locator('#demo-box3-app1').isVisible() &&
      await box3.locator('#demo-box3-app2').isVisible(),
      `${prefix}: clicking Box 3 expands Box 3`);
    await page.waitForFunction(() => [...document.querySelectorAll('#demonstrations img')].every(i => i.complete && i.naturalWidth > 0), null, { timeout: 10000 });
    record(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
      `${prefix}: expanded Box 2 and Box 3 fit viewport without horizontal scroll`);
    await section.screenshot({ path: path.join(OUT, `demonstrations-${file.replace('.html', '')}-${width}.png`) });
    await box2.locator('.demo-box-summary').click();
    await box3.locator('.demo-box-summary').click();
    record(await box2.evaluate(d => !d.open) && await box3.evaluate(d => !d.open) &&
      await box2.locator('.demo-box-body').isHidden() && await box3.locator('.demo-box-body').isHidden(),
      `${prefix}: clicking Box 2 and Box 3 summaries again collapses both modules`);
    await context.close();
  }
}

async function runReadmeWordmarks(browser, origin) {
  for (const name of ['README.md', 'README.zh-CN.md']) {
    const readme = fs.readFileSync(path.resolve(DOCS, '..', '..', name), 'utf8');
    const picture = readme.match(/<picture>[\s\S]*?<\/picture>/)?.[0];
    if (!picture) { record(false, `${name}: themed wordmark present`); continue; }
    for (const colorScheme of ['light', 'dark']) {
      const context = await browser.newContext({ viewport: { width: 700, height: 220 }, colorScheme });
      const page = await context.newPage();
      await page.goto(origin + '/index.html', { waitUntil: 'load' });
      const background = colorScheme === 'dark' ? '#0d1117' : '#ffffff';
      await page.setContent(`<html><head><base href="${origin}/"><style>
        body { margin: 0; min-height: 220px; display: grid; place-items: center; background: ${background}; }
        img { display: block; max-width: 90vw; height: auto; }
        </style></head><body>${picture.replaceAll('docs/site/', '')}</body></html>`);
      const image = page.locator('img');
      await image.evaluate(img => img.decode());
      const actual = await image.evaluate(img => ({ src: new URL(img.currentSrc).pathname,
        width: img.clientWidth, height: img.clientHeight }));
      const suffix = colorScheme === 'dark' ? '-light.svg' : '-wordmark.svg';
      record(actual.src.endsWith(suffix) && actual.width > 0 && actual.height > 0,
        `${name}: ${colorScheme} theme selects the matching transparent vector`, JSON.stringify(actual));
      await page.screenshot({ path: path.join(OUT, `${name}-${colorScheme}.png`) });
      await context.close();
    }
  }
}

async function runLanguages(browser, origin) {
  for (const width of [1440, 1024, 390, 320]) {
    const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce' });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*.mp4', route => route.abort());
    await page.goto(origin + '/SlicerDSpace/index.html#skills', { waitUntil: 'load' });
    await page.locator('.lang-btn[data-lang="zh"]').click();
    await page.waitForURL(url => url.pathname.endsWith('/zh.html'));
    record(new URL(page.url()).hash === '#skills' && await page.locator('html').getAttribute('lang') === 'zh-CN',
      `language ${width}px: Chinese link keeps the current section under a project base path`);
    record(await page.locator('.lang-btn.is-active').getAttribute('data-lang') === 'zh' &&
        await page.locator('#top .btn-primary').innerText() === '观看宣传片' &&
        await page.locator('#skills-title .start-route-text').innerText() === '让 Agent 协助安装、分析与结果检查。' &&
        await page.locator('#skills-title .start-recommended').innerText() === '（推荐）' &&
        await page.locator('.guide-bot__header h2').innerText() === '问问项目助手',
      `language ${width}px: capsule, homepage, Skills and bot use Chinese`);
    const header = await page.evaluate(() => {
      const capsule = document.querySelector('.language-switch').getBoundingClientRect();
      const menu = document.getElementById('menu-btn').getBoundingClientRect();
      const nav = document.querySelector('.site-nav ul').getBoundingClientRect();
      return { fits: capsule.left >= 0 && capsule.right <= innerWidth && capsule.top >= 0,
        separate: innerWidth >= 1024 ? nav.right + 8 <= capsule.left : menu.right + 8 <= capsule.left,
        centered: innerWidth < 1024 || Math.abs((nav.left + nav.right) / 2 - innerWidth / 2) < 1,
        radius: parseFloat(getComputedStyle(document.querySelector('.language-switch')).borderRadius),
        overflow: document.documentElement.scrollWidth > innerWidth };
    });
    record(header.fits && header.separate && header.centered && header.radius >= 999 && !header.overflow,
      `language ${width}px: capsule stays at the right without moving or covering navigation`, JSON.stringify(header));
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.locator('#skills [data-code]').filter({ has: page.locator('#agent-install-prompt') }).locator('.copy').click();
    await page.waitForTimeout(80);
    const copied = (await page.evaluate(() => navigator.clipboard.readText())).replace(/\r\n/g, '\n');
    const expected = (await page.locator('#agent-install-prompt').textContent()).replace(/\r\n/g, '\n');
    record(copied === expected && copied.includes('配置需要的依赖环境'),
      `language ${width}px: copy uses the Chinese Agent prompt`);
    await page.goto(origin + '/SlicerDSpace/zh.html#performance', { waitUntil: 'load' });
    const count = await page.locator('.perf-table .perf-record').count();
    record(count === PERF_COUNT && await page.locator('.perf-table').isVisible() &&
        await page.locator('[data-perf-filter], #perf-count').count() === 0 &&
        await page.locator('#integrated tr[data-method]').count() === 30 &&
        await page.locator('.int-table .method-reference').count() > 35 &&
        await page.locator('#cite').count() === 0,
      `language ${width}px: full translated performance table and complete method/reference catalogs`);
    await page.locator('.lang-btn[data-lang="en"]').click();
    await page.waitForURL(url => url.pathname.endsWith('/index.html'));
    record(new URL(page.url()).hash === '#performance' &&
        await page.locator('html').getAttribute('lang') === 'en' &&
        await page.locator('.lang-btn.is-active').getAttribute('data-lang') === 'en',
      `language ${width}px: English switch restores English and keeps the section`);
    await page.goto(origin + '/zh.html', { waitUntil: 'load' });
    await page.screenshot({ path: path.join(OUT, `site-zh-${width}.png`) });
    record(!errors.length, `language ${width}px: no JavaScript errors`, errors.join('; '));
    await context.close();
  }
  const context = await browser.newContext({ viewport: { width: 390, height: 900 }, javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto(origin + '/index.html', { waitUntil: 'load' });
  await page.locator('.lang-btn[data-lang="zh"]').click();
  await page.waitForURL(url => url.pathname.endsWith('/zh.html'));
  record(await page.locator('html').getAttribute('lang') === 'zh-CN' &&
      await page.locator('#skills-title .start-route-text').innerText() === '让 Agent 协助安装、分析与结果检查。' &&
      await page.locator('#skills-title .start-recommended').innerText() === '（推荐）' &&
      await page.locator('#integrated tr[data-method]').count() === 30,
    'language no JS: Chinese content, commands and all methods are available without scripting');
  record(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    'language no JS: Chinese page fits a mobile viewport');
  await page.locator('.lang-btn[data-lang="en"]').click();
  await page.waitForURL(url => url.pathname.endsWith('/index.html'));
  record(await page.locator('html').getAttribute('lang') === 'en', 'language no JS: English return link works');
  await context.close();
}

// ---------------------------------------------------------------- main
const server = await serve();
const origin = `http://127.0.0.1:${server.address().port}`;
let browser;
try {
  browser = await chromium.launch({
    ...(process.env.CHROME_EXECUTABLE ? { executablePath: process.env.CHROME_EXECUTABLE } : { channel: 'chrome' }),
    headless: true,
    args: ['--autoplay-policy=no-user-gesture-required'],
  });
  if (!process.argv.includes('--performance') && !process.argv.includes('--footer') && !process.argv.includes('--skills')) {
    await runViewport(browser, origin, '1440x900', 1440, 900, {});
    await runViewport(browser, origin, '390x844', 390, 844, {});
    await runViewport(browser, origin, '320x640', 320, 640, { overflowOnly: true });
    // Check header edges around the mobile-navigation breakpoint.
    for (const w of [861, 900, 1023, 1024, 1100]) {
      const ctx = await browser.newContext({ viewport: { width: w, height: 800 } });
      const pg = await ctx.newPage();
      await pg.goto(origin + '/index.html', { waitUntil: 'load' });
      const hdr = await pg.evaluate(() => {
        const inner = document.querySelector('.header-inner');
        const shown = (el) => { const s = getComputedStyle(el), r = el.getBoundingClientRect(); return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0 && r.height > 0; };
        const items = [document.getElementById('menu-btn'),
          ...document.querySelectorAll('#site-nav a, .lang-btn')].filter(shown);
        const rects = items.map((el) => ({ name: (el.textContent || '').trim().slice(0, 20), r: el.getBoundingClientRect() }));
        const hits = [];
        for (let i = 0; i < rects.length; i++) for (let j = i + 1; j < rects.length; j++) {
          const a = rects[i].r, b = rects[j].r;
          if (a.left < b.right - 0.5 && b.left < a.right - 0.5 && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5) hits.push(`${rects[i].name} / ${rects[j].name}`);
        }
        const fits = inner.scrollWidth <= inner.clientWidth && document.documentElement.scrollWidth <= window.innerWidth;
        return { fits, hits };
      });
      record(hdr.fits, `${w}px: header fits and page does not scroll sideways`, 'overflow');
      record(!hdr.hits.length, `${w}px: nav links, language capsule and Menu button do not overlap`, hdr.hits.join('; '));
      await ctx.close();
    }
    await run404AndNoJs(browser, origin);
    await runSkills(browser, origin);
    await runDemonstrations(browser, origin);
    await runLanguages(browser, origin);
    await runReadmeWordmarks(browser, origin);
    await runReferenceLinks(browser, origin);
  }
  if (process.argv.includes('--demonstrations')) await runDemonstrations(browser, origin);
  if (process.argv.includes('--skills')) await runSkills(browser, origin);
  if (!process.argv.includes('--performance') && !process.argv.includes('--skills')) await runFooter(browser, origin);
  if (!process.argv.includes('--footer') && !process.argv.includes('--skills')) {
    for (const width of [1440, 390, 360, 320]) await runPerformance(browser, origin, width);
    await runPerformance(browser, origin, 390, true);
    await runPerformanceFailures(browser, origin);
    await runHomePerformanceNavigation(browser, origin);
  }
} catch (err) {
  record(false, 'browser run', String(err && err.stack || err));
} finally {
  if (browser) await browser.close();
  server.close();
}

const failedCount = results.filter((r) => !r.ok).length;
console.log(`${results.length} checks, ${failedCount} failed`);
console.log(`Screenshots: ${OUT}`);
process.exit(failedCount ? 1 : 0);

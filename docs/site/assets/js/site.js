/* SlicerDSpace project site. Small vanilla script. Everything here is an enhancement:
   the page is complete without it. */
(function () {
  'use strict';

  var chinese = document.documentElement.lang.toLowerCase().indexOf('zh') === 0;
  var labels = chinese ? { play: '播放宣传片', replay: '重新播放', copy: '复制',
    copied: '已复制', manualCopy: '请按 Ctrl+C', film: '（SlicerDSpace 宣传片）' } :
    { play: 'Play film', replay: 'Play again', copy: 'Copy',
      copied: 'Copied', manualCopy: 'Press Ctrl+C', film: ' (SlicerDSpace film)' };

  var reduceMotion = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- Section reveal (fade and rise once) ---------- */
  function initReveal() {
    var items = Array.prototype.slice.call(document.querySelectorAll('.reveal'));
    if (!items.length) { return; }
    if (reduceMotion || !('IntersectionObserver' in window)) {
      items.forEach(function (el) { el.classList.add('in'); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('in');
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
    items.forEach(function (el) { io.observe(el); });
  }

  /* ---------- Mobile menu ---------- */
  function initMenu() {
    var btn = document.getElementById('menu-btn');
    var nav = document.getElementById('site-nav');
    if (!btn || !nav) { return; }

    function setOpen(open, returnFocus) {
      nav.classList.toggle('open', open);
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      if (!open && returnFocus) { btn.focus(); }
    }
    btn.addEventListener('click', function () {
      setOpen(btn.getAttribute('aria-expanded') !== 'true');
    });
    nav.addEventListener('click', function (e) {
      if (e.target.closest && e.target.closest('a')) { setOpen(false); }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && btn.getAttribute('aria-expanded') === 'true') {
        setOpen(false, true);
      }
    });
    /* Leaving the narrow layout closes the menu state. */
    if (window.matchMedia) {
      var mq = window.matchMedia('(min-width: 1024px)');
      var onChange = function () { if (mq.matches) { setOpen(false); } };
      if (mq.addEventListener) { mq.addEventListener('change', onChange); }
      else if (mq.addListener) { mq.addListener(onChange); }
    }
  }

  /* ---------- Current section in the nav ---------- */
  function initScrollSpy() {
    if (!('IntersectionObserver' in window)) { return; }
    var links = Array.prototype.slice.call(document.querySelectorAll('.site-nav li a[href^="#"]'));
    var map = {};
    links.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
    var visible = {};
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) { visible[entry.target.id] = entry.isIntersecting; });
      var current = null;
      Object.keys(map).forEach(function (id) { if (visible[id]) { current = id; } });
      links.forEach(function (a) { a.removeAttribute('aria-current'); });
      if (current) { map[current].setAttribute('aria-current', 'true'); }
    }, { rootMargin: '-35% 0px -55% 0px', threshold: 0 });
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) { io.observe(el); }
    });
  }

  /* ---------- Film: play button ---------- */
  function initFilm() {
    var video = document.getElementById('film-video');
    var frame = document.getElementById('film-frame');
    if (!video || !frame) { return; }

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'film-play';
    btn.innerHTML =
      '<span class="play-circle" aria-hidden="true">' +
      '<svg viewBox="0 0 24 24" width="30" height="30" focusable="false"><path d="M6 3.5v17l14-8.5z" fill="currentColor"></path></svg>' +
      '</span><span class="play-label"></span>';
    var label = btn.querySelector('.play-label');

    function show(text) {
      label.textContent = text;
      btn.setAttribute('aria-label', text + labels.film);
      btn.hidden = false;
    }
    function start() {
      btn.hidden = true;
      video.setAttribute('controls', '');
      var p = video.play();
      if (p && typeof p.catch === 'function') {
        p.catch(function () { /* the native controls stay for a manual start */ });
      }
    }

    video.removeAttribute('controls');
    frame.appendChild(btn);
    show(labels.play);

    btn.addEventListener('click', function () {
      if (video.ended) { video.currentTime = 0; }
      start();
      video.focus({ preventScroll: true });
    });
    video.addEventListener('ended', function () {
      video.removeAttribute('controls');
      show(labels.replay);
    });

    /* Hero button: scroll to the film and start it. */
    var heroBtn = document.querySelector('[data-play-film]');
    if (heroBtn) {
      heroBtn.addEventListener('click', function (e) {
        e.preventDefault();
        var target = document.getElementById('film');
        if (target) {
          target.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
          if (history.replaceState) { history.replaceState(null, '', '#film'); }
        }
        if (video.paused) {
          if (video.ended) { video.currentTime = 0; }
          start();
        }
      });
    }
  }

  /* ---------- Copy buttons ---------- */
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.top = '-1000px';
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
      document.body.removeChild(ta);
      if (ok) { resolve(); } else { reject(new Error('copy failed')); }
    });
  }

  function initCopy() {
    Array.prototype.forEach.call(document.querySelectorAll('[data-code]'), function (box) {
      var btn = box.querySelector('.copy');
      var code = box.querySelector('pre code');
      if (!btn || !code) { return; }
      btn.hidden = false;
      var timer = null;
      btn.addEventListener('click', function () {
        copyText(code.textContent).then(function () {
          btn.textContent = labels.copied;
        }, function () {
          btn.textContent = labels.manualCopy;
          var range = document.createRange();
          range.selectNodeContents(code);
          var sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
        });
        clearTimeout(timer);
        timer = setTimeout(function () { btn.textContent = labels.copy; }, 2000);
      });
    });
  }

  /* ---------- Open any closed <details> containing the fragment target ---------- */
  function openForId(id) {
    if (!id) { return; }
    var target = document.getElementById(id);
    if (!target) { return; }
    var opened = false;
    var parent = target.parentElement ? target.parentElement.closest('details') : null;
    while (parent) {
      if (!parent.open) { parent.open = true; opened = true; }
      parent = parent.parentElement ? parent.parentElement.closest('details') : null;
    }
    if (opened && target.scrollIntoView) { target.scrollIntoView(); }
  }

  function hashId() {
    try { return decodeURIComponent((location.hash || '').slice(1)); } catch (err) { return ''; }
  }

  function initHashOpen() {
    openForId(hashId());
    window.addEventListener('hashchange', function () { openForId(hashId()); });
    /* A link to the hash that is already in the address bar fires no hashchange. */
    document.addEventListener('click', function (e) {
      var link = e.target.closest ? e.target.closest('a[href^="#"]') : null;
      if (link && link.getAttribute('href').length > 1) { openForId(link.getAttribute('href').slice(1)); }
    });
  }

  function initDemoBoxes() {
    Array.prototype.forEach.call(document.querySelectorAll('details.demo-box'), function (box) {
      box.addEventListener('toggle', function () {
        if (!box.open) { return; }
        Array.prototype.forEach.call(box.querySelectorAll('img[loading="lazy"]'), function (img) {
          img.loading = 'eager';
        });
      });
    });
  }

  /* ---------- Language switch: preserve reading position across EN/ZH ---------- */
  var LANG_SCROLL_KEY = 'slicerdspace:lang-scroll';
  var LANG_SCROLL_TTL_MS = 15000;
  var LANG_BLOCK_SELECTOR = 'h1, h2, h3, h4, p, figure, details, table, pre, ul, ol, .card, .perf-panel, .start-route';

  function getHeaderRefY() {
    var header = document.getElementById('site-header');
    return (header ? header.getBoundingClientRect().height : 64) + 12;
  }

  function revealVisibleNow() {
    var vh = window.innerHeight || document.documentElement.clientHeight || 900;
    Array.prototype.forEach.call(document.querySelectorAll('.reveal:not(.in)'), function (el) {
      var rect = el.getBoundingClientRect();
      if (rect.top < vh * 1.1) {
        el.classList.add('in');
      }
    });
  }

  function saveLangScroll() {
    var y = window.scrollY || window.pageYOffset || 0;
    var docEl = document.documentElement;
    var maxScroll = Math.max(0, (docEl.scrollHeight || 0) - (window.innerHeight || 0));
    var openDetails = [];
    Array.prototype.forEach.call(document.querySelectorAll('details'), function (d, idx) {
      if (d.open) { openDetails.push(idx); }
    });
    if (y <= 4 && !openDetails.length) {
      try { sessionStorage.removeItem(LANG_SCROLL_KEY); } catch (err) {}
      return;
    }
    var refY = getHeaderRefY();
    var state = {
      ts: Date.now(),
      scrollY: Math.round(y),
      scrollRatio: maxScroll > 0 ? y / maxScroll : 0,
      atBottom: maxScroll > 0 && y >= maxScroll - 24,
      openDetails: openDetails,
      anchorId: '',
      sectionProgress: 0,
      blockIndex: -1,
      blockOffset: 0
    };
    var main = document.getElementById('main') || document.body;
    var anchors = Array.prototype.slice.call(main.querySelectorAll('section[id], [id]')).filter(function (el) {
      var r = el.getBoundingClientRect();
      return r.height > 0 && r.width > 0;
    });
    var chosen = null;
    for (var i = 0; i < anchors.length; i++) {
      var rect = anchors[i].getBoundingClientRect();
      if (rect.top <= refY + 2) {
        chosen = anchors[i];
      } else if (!chosen) {
        chosen = anchors[i];
        break;
      } else {
        break;
      }
    }
    if (chosen) {
      var cRect = chosen.getBoundingClientRect();
      state.anchorId = chosen.id;
      state.sectionProgress = cRect.height > 0 ? Math.max(0, Math.min(1, (refY - cRect.top) / cRect.height)) : 0;
      if (chosen.id !== 'installer-customizer') {
        var blocks = chosen.querySelectorAll(LANG_BLOCK_SELECTOR);
        var bestIdx = -1;
        var bestTop = -Infinity;
        for (var j = 0; j < blocks.length; j++) {
          var b = blocks[j];
          if (b.closest && b.closest('#installer-customizer')) { continue; }
          var bRect = b.getBoundingClientRect();
          if (bRect.height <= 0 || bRect.width <= 0) { continue; }
          if (bRect.top <= refY + 4 && bRect.top >= bestTop) {
            bestTop = bRect.top;
            bestIdx = j;
          }
        }
        if (bestIdx >= 0) {
          state.blockIndex = bestIdx;
          state.blockOffset = Math.round(bestTop - refY);
        }
      }
    }
    try { sessionStorage.setItem(LANG_SCROLL_KEY, JSON.stringify(state)); } catch (err) {}
  }

  function computeRestoredY(state) {
    var docEl = document.documentElement;
    var maxScroll = Math.max(0, (docEl.scrollHeight || 0) - (window.innerHeight || 0));
    if (state.atBottom) { return maxScroll; }
    var refY = getHeaderRefY();
    var currentY = window.scrollY || window.pageYOffset || 0;
    if (state.anchorId) {
      var anchor = document.getElementById(state.anchorId);
      if (anchor) {
        if (state.blockIndex >= 0 && state.anchorId !== 'installer-customizer') {
          var blocks = anchor.querySelectorAll(LANG_BLOCK_SELECTOR);
          var blk = blocks[state.blockIndex];
          if (blk) {
            var bRect = blk.getBoundingClientRect();
            if (bRect.height > 0 && bRect.width > 0) {
              var byBlock = Math.round(currentY + bRect.top - (refY + state.blockOffset));
              return Math.max(0, Math.min(maxScroll, byBlock));
            }
          }
        }
        var aRect = anchor.getBoundingClientRect();
        if (aRect.height > 0) {
          var byAnchor = Math.round(currentY + aRect.top + state.sectionProgress * aRect.height - refY);
          return Math.max(0, Math.min(maxScroll, byAnchor));
        }
      }
    }
    if (typeof state.scrollRatio === 'number' && maxScroll > 0) {
      return Math.max(0, Math.min(maxScroll, Math.round(state.scrollRatio * maxScroll)));
    }
    return Math.max(0, Math.min(maxScroll, state.scrollY || 0));
  }

  function restoreLangScroll() {
    var raw = null;
    try {
      raw = sessionStorage.getItem(LANG_SCROLL_KEY);
      if (raw !== null) { sessionStorage.removeItem(LANG_SCROLL_KEY); }
    } catch (err) {
      return false;
    }
    if (!raw) { return false; }
    var state;
    try { state = JSON.parse(raw); } catch (err) { return false; }
    if (!state || !state.ts || (Date.now() - state.ts) > LANG_SCROLL_TTL_MS) { return false; }

    if (state.openDetails && state.openDetails.length) {
      var allDetails = document.querySelectorAll('details');
      state.openDetails.forEach(function (idx) {
        if (allDetails[idx]) { allDetails[idx].open = true; }
      });
    }

    var cancelled = false;
    var ro = null;
    var stopTimer = null;

    function stopTracking() {
      if (cancelled) { return; }
      cancelled = true;
      if (ro) { try { ro.disconnect(); } catch (err) {} ro = null; }
      if (stopTimer) { clearTimeout(stopTimer); stopTimer = null; }
      ['wheel', 'touchstart', 'pointerdown', 'keydown', 'hashchange'].forEach(function (ev) {
        window.removeEventListener(ev, stopTracking, true);
      });
    }

    function applyOnce() {
      if (cancelled) { return; }
      var targetY = computeRestoredY(state);
      var docEl = document.documentElement;
      var prevBehavior = docEl.style.scrollBehavior;
      docEl.style.scrollBehavior = 'auto';
      window.scrollTo(0, targetY);
      docEl.style.scrollBehavior = prevBehavior;
      revealVisibleNow();
    }

    ['wheel', 'touchstart', 'pointerdown', 'keydown', 'hashchange'].forEach(function (ev) {
      window.addEventListener(ev, stopTracking, { capture: true, passive: true });
    });

    applyOnce();
    if (window.requestAnimationFrame) {
      window.requestAnimationFrame(applyOnce);
    }
    if (document.readyState !== 'complete') {
      window.addEventListener('load', function () {
        applyOnce();
        if (window.requestAnimationFrame) { window.requestAnimationFrame(applyOnce); }
      }, { once: true });
    }
    if (typeof ResizeObserver === 'function' && document.body) {
      ro = new ResizeObserver(function () { applyOnce(); });
      ro.observe(document.body);
    }
    stopTimer = setTimeout(stopTracking, 1200);
    return true;
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll('.lang-btn'), function (languageLink) {
      languageLink.addEventListener('click', function (event) {
        if (languageLink.classList.contains('is-active')) { event.preventDefault(); return; }
        saveLangScroll();
        var target = new URL(languageLink.href);
        target.hash = location.hash;
        languageLink.href = target.href;
      });
    });
    initMenu();
    initScrollSpy();
    initFilm();
    initCopy();
    initDemoBoxes();
    initHashOpen();
    restoreLangScroll();
    initReveal();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

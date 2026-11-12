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

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll('.lang-btn'), function (languageLink) {
      languageLink.addEventListener('click', function (event) {
        if (languageLink.classList.contains('is-active')) { event.preventDefault(); return; }
        var target = new URL(languageLink.href);
        target.hash = location.hash;
        languageLink.href = target.href;
      });
    });
    initReveal();
    initMenu();
    initScrollSpy();
    initFilm();
    initCopy();
    initDemoBoxes();
    initHashOpen();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

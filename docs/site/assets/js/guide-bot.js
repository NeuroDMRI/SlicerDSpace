/* Local, curated project guide: no network requests, storage, or model service. */
(function () {
  'use strict';
  var bot = document.getElementById('guide-bot');
  if (!bot) { return; }
  var launcher = bot.querySelector('.guide-bot__launcher');
  var panel = bot.querySelector('.guide-bot__panel');
  var backdrop = document.querySelector('.guide-bot__backdrop');
  var header = bot.querySelector('.guide-bot__header');
  var closeButton = bot.querySelector('.guide-bot__close');
  var languageButton = bot.querySelector('.guide-bot__language');
  var messages = bot.querySelector('.guide-bot__messages');
  var form = bot.querySelector('.guide-bot__form');
  var input = form.querySelector('input');
  var siteHeader = document.querySelector('.site-header');
  var floatingPosition = null;
  var starters = Array.prototype.slice.call(bot.querySelectorAll('[data-guide-question]'));
  var language = document.documentElement.lang.toLowerCase().indexOf('zh') === 0 ? 'zh' : 'en';
  var history = [{ type: 'answer', key: 'welcome' }];
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var copy = {
    en: {
      title: 'Ask the guide', description: 'A local guide to the project',
      placeholder: 'Ask about the workflow…', ask: 'Ask', close: 'Close guide',
      open: 'Open SlicerDSpace guide', switchLanguage: 'Switch to Chinese',
      privacy: 'Local answers · Nothing is sent', conversation: 'Guide conversation',
      suggestions: 'Suggested questions', link: 'Explore this section',
      questions: { start: 'Getting started', inputs: 'Inputs', methods: 'Methods',
        dependencies: 'Dependencies', results: 'Results & QC', references: 'References' },
      answers: {
        welcome: { text: 'Hello! I can help you explore SlicerDSpace. Choose a topic or ask a question about the project. I use a small, curated set of answers from this website.' },
        start: { text: 'Install 3D Slicer 5.10.0 from https://download.slicer.org/ (typical paths: /Applications/Slicer.app on macOS, %LOCALAPPDATA%\\slicer.org\\Slicer 5.10.0 on Windows, ~/Slicer-5.10.0-linux-amd64 on Linux), then download the platform archive from https://github.com/NeuroDMRI/SlicerDSpace/releases and install it via View › Extensions Manager › Install Extension from File…. Restart Slicer, open Modules › Diffusion › SlicerDSpace, configure Inputs and Analysis, and click Run SlicerDSpace.', href: '#start' },
        skills: { text: 'We provide an install-and-use Agent Skill. Give its SKILL.md to your local Agent for installation, input preparation, method selection, plan review, requested analyses or batch experiments, and result/QC inspection. Get started includes the Skill and copyable installation and analysis prompts.', href: '#skills' },
        inputs: { text: 'Under Inputs, you can select diffusion MRI in NIfTI (with matching .bval/.bvec), NRRD/NHDR or DICOM folder formats; a compatible brain mask can be supplied or generated. Conversion and validation into the standardised subject directory run when execution starts.', href: '#start' },
        methods: { text: 'The module combines DDSeg, DDParcel and DDSurfer with microstructure, tractography, bundle identification and connectome analyses. DTI is the default microstructure model; use ... beside each stage under Analysis to choose algorithms and adjust stage parameters.', href: '#integrated' },
        dependencies: { text: 'Install SlicerDMRI and PyTorch in Slicer. UKF also needs the separate UKFTractography extension. Other tools, models and licences depend on the selected route; clicking Run SlicerDSpace checks dependencies first and opens Prepare selected analysis environment if anything is missing.', href: '#start' },
        results: { text: 'Completed outputs are recorded for on-demand inspection rather than loaded automatically. In Results, select a run and use the eye icon to load or toggle outputs in the Slicer scene, right-click to unload or open table/JSON files, and click Open report for the HTML QC summary.', href: '#outputs' },
        references: { text: 'Each entry in What is integrated links directly to its primary paper, DOI or official software documentation.', href: '#integrated' },
        gpu: { text: 'Hardware support depends on the selected method and installed backend. Set Auto, CPU or a specific CUDA GPU via Compute device under Analysis; some stages still use CPU tools even on a machine with a GPU.', href: '#start' },
        performance: { text: 'The performance section reports measured comparisons for specific operations and inputs. The results should be interpreted at their recorded scope rather than as a single whole-workflow speedup.', href: '#performance' },
        identity: { text: 'I am a local site guide, inspired by Chengjin Bot. I match questions to curated project information; I am not a live AI model, and your questions stay in this browser tab.' },
        fallback: { text: 'I do not have a reliable answer for that yet. Try a suggested topic, or browse the methods and references on this page.', href: '#integrated' }
      }
    },
    zh: {
      title: '问问项目助手', description: 'SlicerDSpace 本地导览',
      placeholder: '输入有关使用流程的问题…', ask: '提问', close: '关闭助手',
      open: '打开 SlicerDSpace 助手', switchLanguage: 'Switch to English',
      privacy: '本地回答 · 不会发送输入', conversation: '助手对话',
      suggestions: '推荐问题', link: '查看相关内容',
      questions: { start: '如何开始', inputs: '输入数据', methods: '分析方法',
        dependencies: '环境依赖', results: '结果与 QC', references: '参考文献' },
      answers: {
        welcome: { text: '你好！我是 SlicerDSpace 项目助手。可以选择下方问题，或输入有关项目的问题。我根据本网页整理的内容回答。' },
        start: { text: '请先从 https://download.slicer.org/ 安装 3D Slicer 5.10.0（常见路径：macOS 为 /Applications/Slicer.app，Windows 为 %LOCALAPPDATA%\\slicer.org\\Slicer 5.10.0，Linux 为 ~/Slicer-5.10.0-linux-amd64）；随后从 https://github.com/NeuroDMRI/SlicerDSpace/releases 下载对应系统的发布包，在 Slicer 中打开 View › Extensions Manager › Install Extension from File… 完成安装并重启，再于 Modules › Diffusion › SlicerDSpace 中配置 Inputs、Analysis 与运行前检查。', href: '#start' },
        skills: { text: '我们提供安装与使用 Agent Skill。将 SKILL.md 交给自己的本地 Agent，可按指南完成安装、输入准备、方法选择、计划检查、按请求分析或批量实验，以及结果与 QC 检查。Get started 内提供文件入口和可复制的安装、分析指令。', href: '#skills' },
        inputs: { text: '支持 NIfTI（含匹配的 .bval/.bvec）、NRRD/NHDR 或 DICOM 文件夹输入；脑掩膜可自行提供或生成。在 Inputs 下选好数据后，点击运行会自动完成转换并校验至标准化被试目录。', href: '#start' },
        methods: { text: '插件将 DDSeg、DDParcel 和 DDSurfer 与微结构、纤维追踪、纤维束识别及连接组分析结合。微结构默认使用 DTI；可通过 Analysis 各阶段旁的“...”按钮选择算法并配置阶段参数。', href: '#integrated' },
        dependencies: { text: '在 Slicer 中安装 SlicerDMRI 和 PyTorch。UKF 还需要独立的 UKFTractography 扩展。其他工具、模型及许可取决于所选路线；点击 Run SlicerDSpace 时会自动执行运行前检查，缺少依赖时弹出 Prepare selected analysis environment 供确认准备。', href: '#start' },
        results: { text: '运行生成的输出按需加载，不会自动载入 Slicer 场景。请在 Results 中选择运行记录，点击输出项旁的眼睛图标在场景中按需加载或切换显隐（右键可从场景卸载或直接打开表格与 JSON 文件），点击 Open report 查看 HTML QC 报告。', href: '#outputs' },
        references: { text: '已集成的方法表为各条分析路线提供论文、DOI 或官方软件文档的直接链接。', href: '#integrated' },
        gpu: { text: '硬件支持取决于所选方法和已安装的后端。可在 Analysis 下的 Compute device 中选择 Auto、CPU 或指定 CUDA GPU；即使设备有 GPU，部分步骤仍使用 CPU 工具。', href: '#start' },
        performance: { text: '性能部分展示了特定操作和输入条件下的实测对照，应按各自记录的范围理解，不能直接当作完整工作流的统一加速比。', href: '#performance' },
        identity: { text: '我是参考 Chengjin Bot 设计的本地项目助手，根据预设项目资料匹配问题，不是在线大模型；输入只留在当前浏览器页面中。' },
        fallback: { text: '这个问题暂时没有可靠的预设回答。可以试试推荐问题，或浏览页面中的方法与参考文献。', href: '#integrated' }
      }
    }
  };

  function viewport() {
    var v = window.visualViewport;
    return { left: v ? v.offsetLeft : 0, top: v ? v.offsetTop : 0,
      width: v ? v.width : window.innerWidth, height: v ? v.height : window.innerHeight };
  }
  function positionPanel() {
    if (panel.hidden) { return; }
    var v = viewport();
    var mobile = v.width <= 680;
    var headerBottom = siteHeader.getBoundingClientRect().bottom;
    var top = Math.max(v.top, headerBottom) + 12;
    var keyboardOpen = v.height < window.innerHeight * .75;
    var bottomInset = mobile && !keyboardOpen && v.height >= 480 ? 84 : 12;
    var bottom = v.top + v.height - bottomInset;
    if (bottom - top < 180) { top = v.top + 12; }
    var availableHeight = Math.max(0, bottom - top);
    var maxHeight = mobile ? availableHeight : Math.min(v.height * .78, availableHeight);
    var width = Math.max(0, mobile ? v.width - 32 : Math.min(370, v.width - 32));
    panel.style.width = width + 'px';
    panel.style.maxHeight = maxHeight + 'px';
    panel.classList.toggle('is-compact', maxHeight < 360);
    panel.classList.toggle('is-tiny', maxHeight < 240);
    backdrop.hidden = !mobile;
    document.body.classList.toggle('guide-bot-open', mobile);
    if (mobile) { panel.setAttribute('aria-modal', 'true'); }
    else { panel.removeAttribute('aria-modal'); }
    var anchor = bot.getBoundingClientRect();
    var x = v.left + (v.width - width) / 2;
    if (!mobile) {
      x = anchor.right + 16;
      if (x + width > v.left + v.width - 16) { x = anchor.left - width - 16; }
    }
    x = Math.max(v.left + 16, Math.min(x, v.left + v.width - width - 16));
    var preferredY = mobile ? bottom - panel.offsetHeight : anchor.top;
    var y = Math.max(top, Math.min(preferredY, bottom - panel.offsetHeight));
    panel.style.left = (x - anchor.left) + 'px';
    panel.style.top = (y - anchor.top) + 'px';
  }
  function fitDraggedLauncher() {
    if (!floatingPosition) { return; }
    var v = viewport();
    bot.style.left = Math.max(v.left + 8, Math.min(floatingPosition.left, v.left + v.width - 60)) + 'px';
    bot.style.top = Math.max(v.top + 8, Math.min(floatingPosition.top, v.top + v.height - 60)) + 'px';
  }
  function setOpen(open, restoreFocus) {
    panel.hidden = !open;
    bot.classList.toggle('is-open', open);
    launcher.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) {
      var menuButton = document.getElementById('menu-btn');
      if (menuButton && menuButton.getAttribute('aria-expanded') === 'true') { menuButton.click(); }
      positionPanel();
      if (viewport().width <= 680) { closeButton.focus({ preventScroll: true }); }
    } else {
      backdrop.hidden = true;
      document.body.classList.remove('guide-bot-open');
      panel.removeAttribute('aria-modal');
      if (restoreFocus) { launcher.focus({ preventScroll: true }); }
    }
  }
  function render() {
    var t = copy[language];
    bot.querySelector('#guide-bot-title').textContent = t.title;
    bot.querySelector('.guide-bot__description').textContent = t.description;
    bot.querySelector('.guide-bot__privacy').textContent = t.privacy;
    bot.querySelector('.guide-bot__starters').setAttribute('aria-label', t.suggestions);
    bot.querySelector('label').textContent = t.title;
    messages.setAttribute('aria-label', t.conversation);
    input.placeholder = t.placeholder;
    form.querySelector('button').textContent = t.ask;
    closeButton.setAttribute('aria-label', t.close);
    backdrop.setAttribute('aria-label', t.close);
    launcher.setAttribute('aria-label', t.open);
    launcher.title = t.open;
    languageButton.textContent = language === 'en' ? '中文' : 'EN';
    languageButton.setAttribute('aria-label', t.switchLanguage);
    starters.forEach(function (button) { button.textContent = t.questions[button.dataset.guideQuestion]; });
    messages.replaceChildren();
    history.forEach(function (entry) {
      var article = document.createElement('article');
      article.className = 'guide-bot__message' + (entry.type === 'user' ? ' guide-bot__message--user' : '');
      var p = document.createElement('p');
      var answer = entry.type === 'answer' ? t.answers[entry.key] : null;
      p.textContent = answer ? answer.text : (entry.key ? t.questions[entry.key] : entry.text);
      article.appendChild(p);
      if (answer && answer.href) {
        var link = document.createElement('a');
        link.href = answer.href;
        link.textContent = t.link;
        article.appendChild(link);
      }
      messages.appendChild(article);
    });
    messages.scrollTop = messages.scrollHeight;
    positionPanel();
  }
  function answer(key, question, delay) {
    history.push(question);
    render();
    window.setTimeout(function () {
      history.push({ type: 'answer', key: key });
      if (history.length > 81) { history.splice(1, 2); }
      render();
    }, delay);
  }
  function match(text) {
    var q = text.toLowerCase();
    var rules = [
      ['skills', /\b(skills?|skill\.md)\b|技能|agent.*安装|安装.*agent/],
      ['identity', /\b(who are you|what are you|bot|assistant|ai)\b|你是谁|你是(什么|谁)|助手|机器人/],
      ['references', /\b(reference|references|citation|citations|paper|papers|cite)\b|引用|参考文献|论文/],
      ['performance', /\b(performance|speed|benchmark|runtime|optimization)\b|性能|速度|耗时|加速|优化/],
      ['gpu', /\b(gpu|cuda|cpu|graphics card)\b|显卡|算力|显存/],
      ['results', /\b(result|results|output|outputs|quality|qc|inspect)\b|结果|质控|质量|加载|查看/],
      ['dependencies', /\b(dependenc(?:y|ies)|environment|extension|ukf|install tools)\b|依赖|环境|扩展|权重|模型下载/],
      ['start', /\b(install|installation|setup|start|begin|run|first)\b|安装|入门|开始|运行|怎么用|如何用/],
      ['inputs', /\b(input|inputs|dicom|nifti|nrrd|bval|bvec|subject|data)\b|输入|数据|被试|梯度|掩膜/],
      ['methods', /\b(method|methods|algorithm|algorithms|microstructure|tractography|bundles|connectome|dti|noddi|dki|ddseg|ddparcel|ddsurfer|tractcloud|wmql|wma|saf)\b|算法|方法|微结构|纤维|分割|分区|连接组|皮层/]
    ];
    for (var i = 0; i < rules.length; i++) { if (rules[i][1].test(q)) { return rules[i][0]; } }
    return 'fallback';
  }

  var pointer = null;
  var suppressClick = false;
  function startDrag(event) {
    if (event.button !== 0) { return; }
    if (event.currentTarget === header && event.target.closest('button, input, a')) { return; }
    if (event.currentTarget === header) { event.preventDefault(); }
    var rect = bot.getBoundingClientRect();
    pointer = { id: event.pointerId, x: event.clientX, y: event.clientY,
      left: rect.left, top: rect.top, moved: false };
    try { event.currentTarget.setPointerCapture(event.pointerId); } catch (ignore) { /* Browser may reject capture. */ }
  }
  function moveDrag(event) {
    if (!pointer || event.pointerId !== pointer.id) { return; }
    var dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
    if (!pointer.moved && Math.hypot(dx, dy) < 6) { return; }
    pointer.moved = true;
    bot.classList.add('is-dragging');
    var v = viewport();
    floatingPosition = {
      left: Math.max(v.left + 8, Math.min(pointer.left + dx, v.left + v.width - 60)),
      top: Math.max(v.top + 8, Math.min(pointer.top + dy, v.top + v.height - 60))
    };
    fitDraggedLauncher();
    bot.style.bottom = 'auto';
    positionPanel();
  }
  function endDrag(event) {
    if (!pointer || event.pointerId !== pointer.id) { return; }
    suppressClick = pointer.moved;
    pointer = null;
    bot.classList.remove('is-dragging');
    window.setTimeout(function () { suppressClick = false; }, 0);
  }
  [launcher, header].forEach(function (target) {
    target.addEventListener('pointerdown', startDrag);
    target.addEventListener('pointermove', moveDrag);
    target.addEventListener('pointerup', endDrag);
    target.addEventListener('pointercancel', endDrag);
  });
  launcher.addEventListener('click', function () {
    if (suppressClick) { suppressClick = false; return; }
    setOpen(panel.hidden, !panel.hidden);
  });
  closeButton.addEventListener('click', function () { setOpen(false, true); });
  backdrop.addEventListener('click', function () { setOpen(false, true); });
  languageButton.addEventListener('click', function () { language = language === 'en' ? 'zh' : 'en'; render(); });
  starters.forEach(function (button) {
    button.addEventListener('click', function () {
      var key = button.dataset.guideQuestion;
      answer(key, { type: 'user', key: key }, 160);
    });
  });
  form.addEventListener('submit', function (event) {
    event.preventDefault();
    var question = input.value.trim();
    if (!question) { return; }
    if (/[\u3400-\u9fff]/.test(question)) { language = 'zh'; }
    answer(match(question), { type: 'user', text: question }, 180);
    input.value = '';
  });
  panel.addEventListener('click', function (event) {
    if (event.target.closest('a[href^="#"]')) { setOpen(false, false); }
  });
  document.addEventListener('pointerdown', function (event) {
    if (!panel.hidden && viewport().width > 680 && !bot.contains(event.target)) { setOpen(false, false); }
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && !panel.hidden) { setOpen(false, true); }
    else if (event.key === 'Tab' && !panel.hidden && viewport().width <= 680) {
      var controls = Array.prototype.filter.call(panel.querySelectorAll('button, input, a[href]'),
        function (control) { return !control.disabled && control.getClientRects().length; });
      var first = controls[0], last = controls[controls.length - 1];
      if (!first) { return; }
      if (!panel.contains(document.activeElement) || (event.shiftKey && document.activeElement === first) ||
          (!event.shiftKey && document.activeElement === last)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      }
    }
  });
  function resize() { fitDraggedLauncher(); positionPanel(); }
  window.addEventListener('resize', resize);
  var layoutPending = false;
  function scheduleLayout() {
    if (layoutPending) { return; }
    layoutPending = true;
    window.requestAnimationFrame(function () { layoutPending = false; resize(); });
  }
  window.addEventListener('scroll', scheduleLayout, { passive: true });
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', resize);
    window.visualViewport.addEventListener('scroll', resize);
  }
  if (window.ResizeObserver) {
    new ResizeObserver(positionPanel).observe(panel);
    new ResizeObserver(scheduleLayout).observe(siteHeader);
  }
  if (document.fonts && document.fonts.ready) { document.fonts.ready.then(scheduleLayout); }
  if (!reduceMotion) {
    var lastPointer = { x: 0, y: 0 }, eyeTarget = { x: 0, y: 0 }, eyeCurrent = { x: 0, y: 0 };
    var hasPointer = false;
    window.addEventListener('pointermove', function (event) {
      hasPointer = true;
      lastPointer.x = event.clientX;
      lastPointer.y = event.clientY;
    }, { passive: true });
    window.setInterval(function () {
      if (!hasPointer) { return; }
      var rect = launcher.getBoundingClientRect();
      var dx = lastPointer.x - rect.left - rect.width / 2;
      var dy = lastPointer.y - rect.top - rect.height / 2;
      var distance = Math.max(Math.hypot(dx, dy), 1);
      eyeTarget.x = dx / distance * 5;
      eyeTarget.y = dy / distance * 5;
    }, 150);
    function animateEyes() {
      eyeCurrent.x += (eyeTarget.x - eyeCurrent.x) * .05;
      eyeCurrent.y += (eyeTarget.y - eyeCurrent.y) * .05;
      launcher.style.setProperty('--eye-x', eyeCurrent.x + 'px');
      launcher.style.setProperty('--eye-y', eyeCurrent.y + 'px');
      window.requestAnimationFrame(animateEyes);
    }
    window.requestAnimationFrame(animateEyes);
  }
  var today = new Date(), month = today.getMonth() + 1, day = today.getDate();
  bot.classList.toggle('is-festive', (month === 12 && day >= 15) || (month === 1 && day <= 5));
  render();
  launcher.hidden = false;
  resize();
})();

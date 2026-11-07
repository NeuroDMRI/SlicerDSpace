#!/usr/bin/env python3
"""Build a static Chinese homepage using the shared English structure and assets."""

import argparse
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/site/index.html'
TARGET = ROOT / 'docs/site/zh.html'
CATALOG = ROOT / 'docs/website/zh-CN.json'
VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
        'meta', 'param', 'source', 'track', 'wbr', 'path'}
KEEP = {'SlicerDSpace', 'SLICERDSPACE', 'DDSeg', 'DDParcel', 'DDSurfer', 'DTI', 'DKI', 'NODDI',
        'SAF', 'UKF', 'WMA', 'WMQL', 'TractCloud', 'SIFT2', 'MK-Curve', 'MP-PCA',
        'MRtrix3', 'DIPY', 'SlicerDMRI', 'CSD/ACT', 'CSD / MSMT-CSD', 'NODDI / AMICO',
        'MRtrix3 dwi2mask', 'MRtrix3 mtnormalise', 'FSL eddy', 'FSL topup',
        'Slicer BRAINSFit / BRAINSResample',
        'DICOM / NIfTI / NRRD', 'CITATION.cff', 'LICENSE.txt', 'lemo-opuscar', 'ds004666', 'bash', 'Eng'}


class ChinesePage(HTMLParser):
    def __init__(self, translations):
        super().__init__(convert_charrefs=True)
        self.translations = translations
        self.stack = []
        self.output = []
        self.missing = set()

    def translate(self, value):
        key = ' '.join(value.split())
        if not key:
            return value
        translated = self.translations.get(key)
        if key in KEEP:
            return value
        if translated is None:
            number = re.fullmatch(r'([\d.,]+) (min|ms|s)', key)
            reference = re.fullmatch(r'Reference (\d+|\[n\])', key)
            saving = re.fullmatch(r'([\d.]+)% less time', key)
            dated = re.fullmatch(r'(.+) / record ([\d-]+)', key)
            if number:
                translated = number[1] + {'min': ' 分钟', 'ms': ' 毫秒', 's': ' 秒'}[number[2]]
            elif reference:
                translated = '参考文献 ' + reference[1]
            elif saving:
                translated = '耗时减少 ' + saving[1] + '%'
            elif dated and dated[1] in self.translations:
                translated = self.translations[dated[1]] + ' / 记录 ' + dated[2]
        if translated is None:
            if re.search(r'[A-Za-z]{2}', key):
                self.missing.add(key)
            return value
        return value[:len(value) - len(value.lstrip())] + translated + value[len(value.rstrip()):]

    def excluded(self):
        return any(tag in ('script', 'style', 'title', 'svg') or
                   bool({'ref-text', 'method-reference'} & set(attrs.get('class', '').split())) or
                   (tag == 'pre' and not any(a.get('id', '').startswith('agent-') for _, a in self.stack))
                   for tag, attrs in self.stack)

    def handle_decl(self, decl):
        self.output.append('<!' + decl + '>')

    def handle_comment(self, data):
        self.output.append('<!--' + data + '-->')

    def tag(self, tag, attrs, closed=False):
        original = dict(attrs)
        if 'lang-btn' in original.get('class', '').split():
            selected = original.get('data-lang') == 'zh'
            attrs = [(k, ('lang-btn is-active' if selected else 'lang-btn') if k == 'class' else v)
                     for k, v in attrs if k != 'aria-current']
            if selected:
                attrs.append(('aria-current', 'page'))
        values = []
        for name, value in attrs:
            if value is not None:
                if tag == 'html' and name == 'lang':
                    value = 'zh-CN'
                elif 'lang-btn' in original.get('class', '').split() and name == 'aria-label':
                    value = '切换到英文' if original.get('data-lang') == 'en' else '切换到中文'
                elif tag == 'link' and original.get('rel') == 'canonical' and name == 'href':
                    value += 'zh.html'
                elif tag == 'meta' and original.get('property') == 'og:url' and name == 'content':
                    value += 'zh.html'
                elif name == 'title' and 'method-reference' in original.get('class', '').split():
                    pass
                elif name in ('alt', 'aria-label', 'title', 'placeholder'):
                    value = self.translate(value)
                elif tag == 'meta' and name == 'content' and (original.get('name') == 'description' or original.get('property') == 'og:description'):
                    value = self.translate(value)
            values.append(name if value is None else name + '="' + html.escape(value, quote=True) + '"')
        self.output.append('<' + tag + (' ' + ' '.join(values) if values else '') + ('/>' if closed else '>'))
        if not closed and tag not in VOID:
            self.stack.append((tag, original))

    def handle_starttag(self, tag, attrs):
        self.tag(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self.tag(tag, attrs, closed=True)

    def handle_endtag(self, tag):
        self.output.append('</' + tag + '>')
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
        elif tag not in VOID:
            raise ValueError('Unbalanced source HTML: ' + tag)

    def handle_data(self, data):
        if any(tag in ('script', 'style') for tag, _ in self.stack):
            self.output.append(data)
            return
        if not self.excluded():
            # Agent prompts are prose; shell commands and other code stay literal.
            if not any(tag == 'code' and not attrs.get('id', '').startswith('agent-') for tag, attrs in self.stack):
                data = self.translate(data)
        self.output.append(html.escape(data, quote=False))


def render(translations=None):
    if translations is None:
        translations = json.loads(CATALOG.read_text(encoding='utf-8'))
    parser = ChinesePage(translations)
    parser.feed(SOURCE.read_text(encoding='utf-8'))
    parser.close()
    if parser.stack:
        raise ValueError('Unclosed source HTML tags')
    return ''.join(parser.output), sorted(parser.missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--inventory', action='store_true', help='List text still awaiting translation')
    args = parser.parse_args()
    page, missing = render()
    if args.inventory:
        print(json.dumps(missing, ensure_ascii=False, indent=2))
        return
    if missing:
        raise SystemExit('Missing Chinese translations:\n' + '\n'.join(missing))
    if args.check:
        if not TARGET.is_file() or TARGET.read_text(encoding='utf-8') != page:
            raise SystemExit('Chinese homepage is stale; run scripts/build_website_locales.py')
    else:
        TARGET.write_text(page, encoding='utf-8')
    print('PASS static Chinese homepage, translations, shared structure and assets')


if __name__ == '__main__':
    main()

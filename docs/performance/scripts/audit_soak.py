#!/usr/bin/env python3
"""Audit every soak response; validate Bench normalization against Rust perf artifacts."""
import argparse
import collections
import functools
import json
import pathlib
import unicodedata

DIGITS = {c: n for n, chars in enumerate(['零〇', '一壹', '二两贰', '三叁', '四肆', '五伍', '六陆', '七柒', '八捌', '九玖']) for c in chars}
SMALL = {'十': 10, '百': 100, '千': 1000}
LARGE = {'万': 10000, '亿': 100000000}


def normalize(text):
    chars = ''.join(' ' if c == '\u3000' else chr(ord(c) - 0xfee0) if '\uff01' <= c <= '\uff5e' else c for c in text).lower()
    result = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isspace() or unicodedata.category(c).startswith('P'):
            i += 1
            continue
        end = i
        while end < len(chars) and chars[end] in DIGITS | SMALL | LARGE:
            end += 1
        if end > i:
            group = chars[i:end]
            if all(c in DIGITS for c in group):
                result.append(''.join(str(DIGITS[c]) for c in group))
            else:
                total = section = number = 0
                for c in group:
                    if c in DIGITS:
                        number = DIGITS[c]
                    elif c in SMALL:
                        section += max(number, 1) * SMALL[c]
                        number = 0
                    else:
                        section += number
                        total += max(section, 1) * LARGE[c]
                        section = number = 0
                result.append(str(total + section + number))
            i = end
            continue
        def cjk(ch):
            return '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf'
        if c == '儿' and i > 0 and cjk(chars[i-1]) and (i+1 == len(chars) or not cjk(chars[i+1]) or chars[i+1] in '啊呀呢吧吗嘛啦了的'):
            i += 1
            continue
        result.append(c)
        i += 1
    return ''.join(result)


@functools.lru_cache(maxsize=None)
def distance(a, b):
    # Myers bit-vector Levenshtein. Python integers cover arbitrary string lengths.
    if not a:
        return len(b)
    matches = {}
    for i, c in enumerate(a):
        matches[c] = matches.get(c, 0) | (1 << i)
    positive, negative, score, top = ~0, 0, len(a), 1 << (len(a)-1)
    for c in b:
        equal = matches.get(c, 0)
        xv = equal | negative
        xh = (((equal & positive) + positive) ^ positive) | equal
        ph = negative | ~(xh | positive)
        mh = positive & xh
        score += bool(ph & top) - bool(mh & top)
        ph = (ph << 1) | 1
        mh <<= 1
        positive = mh | ~(xv | ph)
        negative = ph & xv
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--validation-dir', required=True)
    parser.add_argument('--soak-dir', required=True)
    args = parser.parse_args()
    validated = 0
    for path in pathlib.Path(args.validation_dir).glob('*.json'):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        quality = data.get('cer', data)
        if not isinstance(quality, dict) or not isinstance(quality.get('items'), list):
            continue
        for item in quality['items']:
            if not isinstance(item, dict) or 'reference_normalized' not in item:
                continue
            assert normalize(item['reference']) == item['reference_normalized'], (str(path), item['index'], 'reference')
            assert normalize(item['hypothesis']) == item['hypothesis_normalized'], (str(path), item['index'], 'hypothesis')
            assert distance(item['reference_normalized'], item['hypothesis_normalized']) == item['substitutions'] + item['insertions'] + item['deletions'], (str(path), item['index'], 'distance')
            validated += 1
    assert validated >= 99
    base = json.loads(pathlib.Path(args.baseline).read_text())
    refs = {item['index']: item for item in base['cer']['items']}
    folder = pathlib.Path(args.soak_dir)
    counts = collections.Counter()
    errors = []
    total_chars = total_edits = baseline_edits = 0
    differences = 0
    short_count = 0
    short_latencies = []
    per_item = collections.defaultdict(lambda: {'runs': 0, 'edits': 0, 'reference_chars': 0, 'min_cer': 10, 'max_cer': 0})
    for line in (folder / 'responses.jsonl').read_text().splitlines():
        row = json.loads(line)
        if not row['ok']:
            errors.append(row.get('error', 'request failed'))
            continue
        data = row['response']['data']
        if len(data) != 1 or data[0]['is_bad']:
            errors.append('bad or missing response')
            continue
        hypothesis = normalize(data[0]['text']['mono'])
        if row['short']:
            short_count += 1
            short_latencies.append(row['latency_sec'])
            if hypothesis != normalize('我要修改商品帖子的价格'):
                errors.append('short reference mismatch')
            continue
        item = refs[row['index']]
        assert row['url'] == item['url']
        counts[row['index']] += 1
        edits = distance(item['reference_normalized'], hypothesis)
        n = len(item['reference_normalized'])
        total_chars += n
        total_edits += edits
        baseline_edits += item['substitutions'] + item['insertions'] + item['deletions']
        differences += hypothesis != item['hypothesis_normalized']
        if not hypothesis or edits > n:
            errors.append(f"empty or CER above 100%, index={row['index']}")
        entry = per_item[row['index']]
        entry['runs'] += 1
        entry['edits'] += edits
        entry['reference_chars'] += n
        entry['min_cer'] = min(entry['min_cer'], edits/n)
        entry['max_cer'] = max(entry['max_cer'], edits/n)
    summary = json.loads((folder / 'summary.json').read_text())
    assert sum(counts.values()) == summary['scheduled'], 'missing completion'
    assert len(counts) == 99, 'not all dataset items covered'
    result = {'validated_rust_items': validated, 'audited_responses': sum(counts.values()), 'short_probes': short_count, 'short_max_latency_sec': max(short_latencies, default=None), 'sampled_weighted_cer': total_edits/total_chars, 'same_input_multiplicity_baseline_cer': baseline_edits/total_chars, 'normalized_transcripts_different_from_baseline': differences, 'min_repeats_per_item': min(counts.values()), 'max_repeats_per_item': max(counts.values()), 'errors': errors, 'per_item': dict(per_item)}
    (folder / 'quality.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'per_item'}, ensure_ascii=False, indent=2))
    assert not errors
    assert total_edits/total_chars <= baseline_edits/total_chars + .001, 'CER exceeds 0.1 percentage point screening tolerance'


if __name__ == '__main__':
    main()

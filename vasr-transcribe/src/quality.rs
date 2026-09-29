/// Return a terse diagnostic when one non-trivial character sequence dominates
/// an output through consecutive repetition. This is a benchmark guard, not a
/// language-quality score.
pub(crate) fn detect_repeated_pattern(text: &str) -> Option<String> {
    let chars = text
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<Vec<_>>();
    if chars.len() < 24 {
        return None;
    }

    let mut best: Option<(usize, usize, usize)> = None;
    for width in 2..=12.min(chars.len() / 5) {
        for start in 0..=chars.len().saturating_sub(width * 5) {
            let pattern = &chars[start..start + width];
            if pattern.iter().all(|ch| *ch == pattern[0]) {
                continue;
            }
            let mut repetitions = 1usize;
            let mut next = start + width;
            while next + width <= chars.len() && chars[next..next + width] == *pattern {
                repetitions += 1;
                next += width;
            }
            let covered = repetitions * width;
            if repetitions >= 5
                && covered >= 24
                && covered.saturating_mul(4) >= chars.len()
                && best.is_none_or(|(best_covered, best_repetitions, _)| {
                    (covered, repetitions) > (best_covered, best_repetitions)
                })
            {
                best = Some((covered, repetitions, width));
            }
        }
    }

    best.map(|(_, repetitions, width)| {
        format!("suspected repeated {width}-character pattern ({repetitions} repeats)")
    })
}

#[cfg(test)]
mod tests {
    use super::detect_repeated_pattern;

    #[test]
    fn flags_a_long_repeated_substring() {
        assert!(detect_repeated_pattern("hello ibilibilibilibilibilibilibilibil end").is_some());
        assert!(
            detect_repeated_pattern("秧undryfaoknryfaoknryfaoknryfaoknryfaoknryfaoknry").is_some()
        );
    }

    #[test]
    fn does_not_flag_short_or_natural_repetitions() {
        assert!(detect_repeated_pattern("你好你好，今天怎么样？").is_none());
        assert!(detect_repeated_pattern("ha ha ha ha ha ha").is_none());
    }
}

//! Lightweight Markdown-to-plain-text stripping for TTS input.
//!
//! LLM replies are often emitted in Markdown. Piper reads literal symbols
//! (e.g. `###`, `*`, `**`) aloud, so before synthesizing speech we strip the
//! markup down to readable plain text. The plain text is only used for audio;
//! the terminal keeps the original formatted reply.

/// Common Markdown emphasis/structural punctuation with no spoken value
/// unless used as meaningful text. We remove styling characters while keeping
/// the words between/after them.
const STRIP_CHARS: &[char] = &['*', '_', '`', '~', '^', '#', '|', '-', '>'];

/// Strip Markdown formatting from `src`, returning readable plain text.
pub fn strip_markdown(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut first = true;

    for line in src.lines() {
        if !first {
            out.push('\n');
        }
        first = false;
        out.push_str(&clean_line(line));
    }
    out
}

/// Clean a single line: drop list bullets/markup prefixes and inline styling.
fn clean_line(line: &str) -> String {
    let trimmed = line.trim_start();

    // Headings: "### foo" -> "foo"
    if let Some(rest) = trimmed.strip_prefix('#') {
        let rest = rest.trim_start();
        return if rest.is_empty() {
            String::new()
        } else {
            strip_inline_markers(rest)
        };
    }

    // List items / quotes: "- foo", "* foo", "> foo", "+ foo"
    if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
        .or_else(|| trimmed.strip_prefix("> "))
    {
        return strip_inline_markers(rest);
    }

    // Horizontal rules "---", "***" -> empty
    if trimmed.chars().all(|c| matches!(c, '-' | '*' | '_' | ' ')) && trimmed.len() >= 3 {
        return String::new();
    }

    // Inline links "[text](url)" -> "text"
    strip_inline_markers(trimmed)
}

/// Remove inline emphasis/backtick markers and link syntax.
fn strip_inline_markers(s: &str) -> String {
    let mut result = String::with_capacity(s.len());

    // Handle inline links/autolinks: [label](url), <url>, ![alt](url)
    let mut rest = s;
    while !rest.is_empty() {
        if let Some(open) = rest.find('[') {
            result.push_str(&rest[..open]);
            if let Some(close_rel) = rest[open..].find(']') {
                let close = open + close_rel;
                // If followed by a "(...)", it's a link -> keep label only.
                if rest[close + 1..].starts_with('(') {
                    let label = &rest[open + 1..close];
                    result.push_str(label);
                    let after = rest[close + 1..].find(')');
                    rest = match after {
                        Some(a) => &rest[close + 1 + a + 1..],
                        None => "",
                    };
                } else {
                    // Bare brackets: keep inner text.
                    result.push_str(&rest[open + 1..close]);
                    rest = &rest[close + 1..];
                }
            } else {
                result.push_str(&rest[open..]);
                rest = "";
            }
            continue;
        }
        result.push_str(rest);
        rest = "";
    }

    // Remove styling characters that carry no spoken value.
    let stripped: String = result
        .chars()
        .filter(|c| !STRIP_CHARS.contains(c))
        .collect();

    // Collapse runs of whitespace to single spaces.
    let mut out = String::with_capacity(stripped.len());
    let mut space = false;
    for c in stripped.chars() {
        if c.is_whitespace() {
            if !space {
                out.push(' ');
            }
            space = true;
        } else {
            out.push(c);
            space = false;
        }
    }
    let out = out.trim();
    out.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_headings() {
        assert_eq!(strip_markdown("### Hello world"), "Hello world");
    }

    #[test]
    fn strips_bold_italics() {
        assert_eq!(strip_markdown("**bold** and *italic*"), "bold and italic");
    }

    #[test]
    fn strips_list_items() {
        assert_eq!(strip_markdown("- one\n- two"), "one\ntwo");
    }

    #[test]
    fn strips_inline_links() {
        assert_eq!(
            strip_markdown("See [here](https://example.com)!"),
            "See here!"
        );
    }

    #[test]
    fn strips_code_backticks() {
        assert_eq!(strip_markdown("run `cargo build`"), "run cargo build");
    }

    #[test]
    fn drops_horizontal_rules() {
        assert_eq!(strip_markdown("a\n---\nb"), "a\n\nb");
    }
}

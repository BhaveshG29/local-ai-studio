//! Safe Markdown rendering for model output.
//!
//! * Raw HTML in model output is shown as text (never interpreted).
//! * Only `http`, `https` and `mailto` links are kept; images are shown as links (remote
//!   images would leak requests and are blocked by the CSP anyway).
//! * Code blocks are highlighted server-side with syntect (CSS classes, themed per UI theme)
//!   and get a one-click "select all" area plus a download link — copying to the clipboard
//!   needs JavaScript, which this application deliberately does not use.
//! * `<think>…</think>` reasoning sections become collapsible blocks.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{LazyLock, Mutex};

use pulldown_cmark::{CodeBlockKind, CowStr, Event, Options, Parser, Tag, TagEnd};
use syntect::html::{ClassStyle, ClassedHTMLGenerator};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

use crate::util::html_escape;

static SYNTAXES: LazyLock<SyntaxSet> = LazyLock::new(SyntaxSet::load_defaults_newlines);
static CACHE: LazyLock<Mutex<HashMap<u64, String>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

const MAX_HIGHLIGHT_BYTES: usize = 60_000;

fn file_extension(lang: &str) -> &str {
    match lang {
        "rust" | "rs" => "rs",
        "python" | "py" => "py",
        "javascript" | "js" => "js",
        "typescript" | "ts" => "ts",
        "bash" | "sh" | "shell" | "zsh" => "sh",
        "c" => "c",
        "cpp" | "c++" => "cpp",
        "java" => "java",
        "go" => "go",
        "html" => "html",
        "css" => "css",
        "json" => "json",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "sql" => "sql",
        "markdown" | "md" => "md",
        _ => "txt",
    }
}

/// Highlights one code block. Falls back to escaped plain text.
pub fn highlight_code(code: &str, lang: &str) -> String {
    let lang = lang.trim().to_lowercase();
    let syntax = if lang.is_empty() || code.len() > MAX_HIGHLIGHT_BYTES {
        None
    } else {
        SYNTAXES.find_syntax_by_token(&lang).or_else(|| SYNTAXES.find_syntax_by_extension(file_extension(&lang)))
    };
    if let Some(syntax) = syntax {
        let mut generator =
            ClassedHTMLGenerator::new_with_class_style(syntax, &SYNTAXES, ClassStyle::SpacedPrefixed { prefix: "hl-" });
        let mut ok = true;
        for line in LinesWithEndings::from(code) {
            if generator.parse_html_for_line_which_includes_newline(line).is_err() {
                ok = false;
                break;
            }
        }
        if ok {
            return generator.finalize();
        }
    }
    html_escape(code)
}

fn code_block_html(code: &str, lang: &str) -> String {
    let label = if lang.is_empty() { "text" } else { lang };
    let file = format!("snippet.{}", file_extension(&lang.to_lowercase()));
    let href = format!(
        "data:text/plain;charset=utf-8,{}",
        percent_encoding::utf8_percent_encode(code, percent_encoding::NON_ALPHANUMERIC)
    );
    format!(
        "<figure class=\"code\"><figcaption><span class=\"code-lang\">{}</span><span class=\"code-hint\">Click the code to select it, then press Ctrl/⌘+C</span><a class=\"code-dl\" href=\"{}\" download=\"{}\">Download</a></figcaption><pre><code class=\"code-select\" tabindex=\"0\">{}</code></pre></figure>",
        html_escape(label),
        html_escape(&href),
        html_escape(&file),
        highlight_code(code, lang)
    )
}

fn safe_url(url: &str) -> bool {
    let u = url.trim().to_lowercase();
    u.starts_with("http://") || u.starts_with("https://") || u.starts_with("mailto:") || u.starts_with('#')
}

fn render_markdown_fragment(src: &str) -> String {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TASKLISTS);
    opts.insert(Options::ENABLE_MATH);
    let src = crate::math::normalize_delimiters(src);
    let parser = Parser::new_ext(&src, opts);
    let mut events: Vec<Event<'_>> = Vec::new();
    let mut in_code: Option<(String, String)> = None;
    let mut image_depth = 0usize;
    for ev in parser {
        if let Some((lang, buf)) = in_code.as_mut() {
            match ev {
                Event::Text(t) => buf.push_str(&t),
                Event::End(TagEnd::CodeBlock) => {
                    let html = code_block_html(buf, lang);
                    events.push(Event::Html(CowStr::from(html)));
                    in_code = None;
                }
                _ => {}
            }
            continue;
        }
        match ev {
            Event::Start(Tag::CodeBlock(kind)) => {
                let lang = match kind {
                    CodeBlockKind::Fenced(l) => l.split_whitespace().next().unwrap_or("").to_string(),
                    CodeBlockKind::Indented => String::new(),
                };
                in_code = Some((lang, String::new()));
            }
            Event::Html(h) | Event::InlineHtml(h) => events.push(Event::Text(h)),
            // Maths becomes MathML (rendered by the browser); TeX the converter doesn't know
            // stays visible as source.
            Event::InlineMath(tex) => match crate::math::to_mathml(&tex, false) {
                Some(m) => events.push(Event::InlineHtml(CowStr::from(m))),
                None => events.push(Event::InlineMath(tex)),
            },
            Event::DisplayMath(tex) => match crate::math::to_mathml(&tex, true) {
                Some(m) => {
                    events.push(Event::InlineHtml(CowStr::from(format!("<span class=\"math-block\">{m}</span>"))))
                }
                None => events.push(Event::DisplayMath(tex)),
            },
            Event::Start(Tag::Link { link_type, dest_url, title, id }) => {
                let dest = if safe_url(&dest_url) { dest_url } else { CowStr::from("#") };
                events.push(Event::Start(Tag::Link { link_type, dest_url: dest, title, id }));
            }
            Event::Start(Tag::Image { dest_url, .. }) => {
                image_depth += 1;
                if safe_url(&dest_url) {
                    events.push(Event::Html(CowStr::from(format!(
                        "<a class=\"md-image-link\" href=\"{}\" rel=\"noopener noreferrer\">[image: ",
                        html_escape(&dest_url)
                    ))));
                } else {
                    events.push(Event::Text(CowStr::from("[image: ")));
                }
            }
            Event::End(TagEnd::Image) => {
                image_depth = image_depth.saturating_sub(1);
                events.push(Event::Html(CowStr::from("]</a>")));
            }
            other => events.push(other),
        }
    }
    if let Some((lang, buf)) = in_code {
        // Unterminated fence (e.g. a response that was cut off).
        events.push(Event::Html(CowStr::from(code_block_html(&buf, &lang))));
    }
    let _ = image_depth;
    let mut out = String::new();
    pulldown_cmark::html::push_html(&mut out, events.into_iter());
    out
}

/// Renders a model response (Markdown, with optional `<think>` sections) to safe HTML.
pub fn markdown_to_html(src: &str) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut h);
    let key = h.finish();
    if let Some(hit) = CACHE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(&key) {
        return hit.clone();
    }
    let mut out = String::new();
    let mut rest = src;
    loop {
        match rest.find("<think>") {
            Some(start) => {
                out.push_str(&render_markdown_fragment(&rest[..start]));
                let after = &rest[start + 7..];
                let (inner, tail, closed) = match after.find("</think>") {
                    Some(end) => (&after[..end], &after[end + 8..], true),
                    None => (after, "", false),
                };
                if !inner.trim().is_empty() {
                    out.push_str(&format!(
                        "<details class=\"reasoning\"{}><summary>{}</summary><div class=\"reasoning-body\">{}</div></details>",
                        if closed { "" } else { " open" },
                        if closed { "Reasoning" } else { "Reasoning (unfinished)" },
                        render_markdown_fragment(inner)
                    ));
                }
                rest = tail;
            }
            None => {
                out.push_str(&render_markdown_fragment(rest));
                break;
            }
        }
    }
    let mut cache = CACHE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if cache.len() > 2000 {
        cache.clear();
    }
    cache.insert(key, out.clone());
    out
}

/// Renders a model card (README) excerpt safely, limited in length.
pub fn model_card_html(src: &str, max_chars: usize) -> String {
    let cut = if src.chars().count() > max_chars {
        let mut s: String = src.chars().take(max_chars).collect();
        // Avoid cutting inside a code fence.
        if s.matches("```").count() % 2 == 1 {
            s.push_str("\n```");
        }
        s.push_str("\n\n…");
        s
    } else {
        src.to_string()
    };
    render_markdown_fragment(&cut)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_html_is_escaped() {
        let html = markdown_to_html("Hello <script>alert(1)</script> <img src=x onerror=alert(1)>");
        assert!(!html.contains("<script>"));
        assert!(!html.contains("<img"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn javascript_links_are_neutralised() {
        let html = markdown_to_html("[click](javascript:alert(1)) and [ok](https://example.com)");
        assert!(!html.contains("javascript:"));
        assert!(html.contains("href=\"https://example.com\""));
    }

    #[test]
    fn images_become_links() {
        let html = markdown_to_html("![cat](https://example.com/cat.png)");
        assert!(!html.contains("<img"));
        assert!(html.contains("[image: cat]"));
    }

    #[test]
    fn code_blocks_are_highlighted_with_classes() {
        let html = markdown_to_html("```rust\nfn main() { let x = 1; }\n```");
        assert!(html.contains("class=\"code\""));
        assert!(html.contains("hl-"));
        assert!(html.contains("download=\"snippet.rs\""));
        assert!(html.contains("code-select"));
    }

    #[test]
    fn unterminated_code_fence_still_renders() {
        let html = markdown_to_html("text\n```python\nprint('hi')");
        assert!(html.contains("print"));
        assert!(html.contains("class=\"code\""));
    }

    #[test]
    fn think_blocks_collapse() {
        let html = markdown_to_html("<think>Let me reason.</think>\n\nThe answer is **42**.");
        assert!(html.contains("<details class=\"reasoning\">"));
        assert!(html.contains("Let me reason."));
        assert!(html.contains("<strong>42</strong>"));
        let open = markdown_to_html("<think>still thinking");
        assert!(open.contains("open"));
    }

    #[test]
    fn tables_and_lists_render() {
        let html = markdown_to_html("| a | b |\n|---|---|\n| 1 | 2 |\n\n- [x] done");
        assert!(html.contains("<table>"));
        assert!(html.contains("checkbox"));
    }
}

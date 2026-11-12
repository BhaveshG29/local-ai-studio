//! A small TeX-to-MathML converter for the maths in model answers.
//!
//! Browsers render MathML natively, so formulas look like formulas without any JavaScript.
//! The converter covers the notation models commonly write — fractions, roots, sub- and
//! superscripts, Greek letters, operators and relations, big operators with limits, function
//! names, accents, font styles (`\mathbb`, `\mathcal`, `\mathbf`, …), `\left…\right`, text,
//! spacing, matrices, `cases` and `aligned`. Anything it does not understand makes
//! [`to_mathml`] return `None`, and the caller shows the TeX source instead of a wrong formula.
//!
//! The output only contains MathML elements built here, and all text from the input is
//! escaped, so it is safe to insert into a page.

use std::fmt::Write as _;

/// Converts TeX maths to a `<math>` element, or `None` when the input uses something this
/// converter does not support (or is unbalanced, or unreasonably large).
pub fn to_mathml(tex: &str, display: bool) -> Option<String> {
    if tex.len() > 8_000 {
        return None;
    }
    let mut p = Parser { chars: tex.chars().collect(), pos: 0, depth: 0, font: None };
    let nodes = p.expr(Stop::End)?;
    if p.pos < p.chars.len() {
        return None;
    }
    let mut out = String::with_capacity(tex.len() * 8);
    out.push_str(if display { "<math display=\"block\">" } else { "<math>" });
    // Keep the source for copy-paste and screen readers that prefer it.
    let _ = write!(out, "<semantics>");
    render(&Node::Row(nodes), &mut out);
    let _ = write!(out, "<annotation encoding=\"application/x-tex\">{}</annotation>", esc(tex.trim()));
    out.push_str("</semantics></math>");
    Some(out)
}

#[derive(Clone, Debug)]
enum Node {
    /// Identifier; `upright` keeps single letters from being italicised.
    Mi(String, bool),
    Mn(String),
    Mo(String, Op),
    Text(String, bool),
    Row(Vec<Node>),
    Frac(Box<Node>, Box<Node>, bool),
    Sqrt(Box<Node>),
    Root(Box<Node>, Box<Node>),
    Scripts {
        base: Box<Node>,
        sub: Option<Box<Node>>,
        sup: Option<Box<Node>>,
        limits: bool,
    },
    Over(Box<Node>, Box<Node>, bool),
    Under(Box<Node>, Box<Node>, bool),
    Table {
        rows: Vec<Vec<Node>>,
        align: Align,
    },
    Space(&'static str),
    Phantom(Box<Node>),
    Boxed(Box<Node>),
    Style(Box<Node>, bool),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Plain,
    /// Does not stretch (TeX's plain `(`).
    Fixed,
    /// `\left(`, `\right)`: stretches to the content.
    Fence,
    /// `\sum`, `\int`: large in display style.
    Large,
    /// Accent mark drawn over or under a base.
    Accent,
    Stretchy,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Align {
    Center,
    Left,
    /// `aligned`: right, left, right, left…
    Pairs,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Font {
    Upright,
    Bold,
    Italic,
    BoldItalic,
    DoubleStruck,
    Script,
    Fraktur,
    Sans,
    Mono,
}

#[derive(Clone, Copy, PartialEq)]
enum Stop {
    End,
    Brace,
    Right,
    Env,
    Bracket,
}

/// How deeply groups and commands may nest. Real formulas stay far below this; each level
/// costs a few kilobytes of stack.
const MAX_DEPTH: usize = 100;

struct Parser {
    chars: Vec<char>,
    pos: usize,
    depth: usize,
    font: Option<Font>,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn skip_space(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.pos += 1;
        }
    }

    /// Reads a command name after a backslash (letters, or one other character).
    fn command(&mut self) -> Option<String> {
        if self.peek() != Some('\\') {
            return None;
        }
        self.pos += 1;
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
            self.pos += 1;
        }
        if self.pos == start {
            let c = self.peek()?;
            self.pos += 1;
            return Some(c.to_string());
        }
        let name: String = self.chars[start..self.pos].iter().collect();
        // A trailing `*` belongs to some names (`\operatorname*`).
        if self.peek() == Some('*') && matches!(name.as_str(), "operatorname") {
            self.pos += 1;
            return Some(format!("{name}*"));
        }
        Some(name)
    }

    fn peek_command(&self) -> Option<String> {
        let mut p = Parser { chars: Vec::new(), pos: 0, depth: 0, font: None };
        p.chars = self.chars[self.pos..self.chars.len().min(self.pos + 32)].to_vec();
        p.command()
    }

    /// A `{…}` group, or a single token, as one node.
    fn arg(&mut self) -> Option<Node> {
        self.skip_space();
        match self.peek()? {
            '{' => {
                self.pos += 1;
                let nodes = self.expr(Stop::Brace)?;
                Some(Node::Row(nodes))
            }
            '\\' => self.atom(),
            '}' | '^' | '_' | '&' => None,
            c => {
                self.pos += 1;
                Some(self.char_node(c))
            }
        }
    }

    /// Raw text of a `{…}` group (for `\text`, `\operatorname`, `\begin`).
    fn raw_group(&mut self) -> Option<String> {
        self.skip_space();
        if self.peek()? != '{' {
            return None;
        }
        self.pos += 1;
        let mut depth = 1;
        let mut s = String::new();
        while let Some(c) = self.peek() {
            self.pos += 1;
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(s);
                    }
                }
                '\\' => {
                    // Spacing commands become spaces; escaped characters (`\{`, `\_`) themselves.
                    if let Some(n) = self.peek() {
                        self.pos += 1;
                        match n {
                            ',' | ':' | ';' | ' ' | '>' => s.push(' '),
                            '!' => {}
                            _ => {
                                if n.is_ascii_alphabetic() {
                                    s.push('\\');
                                }
                                s.push(n);
                            }
                        }
                    }
                    continue;
                }
                _ => {}
            }
            s.push(c);
        }
        None
    }

    fn optional_bracket(&mut self) -> Option<Option<Node>> {
        self.skip_space();
        if self.peek() != Some('[') {
            return Some(None);
        }
        self.pos += 1;
        let nodes = self.expr(Stop::Bracket)?;
        Some(Some(Node::Row(nodes)))
    }

    fn expr(&mut self, stop: Stop) -> Option<Vec<Node>> {
        self.nested(|p| p.expr_inner(stop))
    }

    /// Runs `f` one nesting level deeper, refusing input nested beyond [`MAX_DEPTH`]: the
    /// parser is recursive, and unbounded input would overflow the stack, which aborts the
    /// whole app (a stack overflow can't be caught like a panic).
    fn nested<T>(&mut self, f: impl FnOnce(&mut Self) -> Option<T>) -> Option<T> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let out = f(self);
        self.depth -= 1;
        out
    }

    fn expr_inner(&mut self, stop: Stop) -> Option<Vec<Node>> {
        let mut out: Vec<Node> = Vec::new();
        loop {
            self.skip_space();
            let Some(c) = self.peek() else {
                if stop == Stop::End {
                    break;
                }
                return None;
            };
            match c {
                '}' => {
                    if stop == Stop::Brace {
                        self.pos += 1;
                        break;
                    }
                    return None;
                }
                ']' if stop == Stop::Bracket => {
                    self.pos += 1;
                    break;
                }
                '&' => {
                    if stop == Stop::Env {
                        break;
                    }
                    return None;
                }
                '^' | '_' | '\'' => {
                    // Scripts without a base (`^2`, `_i` at the start): attach to an empty base.
                    let base = out.pop().unwrap_or(Node::Row(Vec::new()));
                    let node = self.scripts(base)?;
                    out.push(node);
                }
                '\\' => {
                    match self.peek_command().as_deref() {
                        Some("right") if stop == Stop::Right => break,
                        Some("end") if stop == Stop::Env => break,
                        Some("\\") if stop == Stop::Env => break,
                        Some("right" | "end") => return None,
                        Some("\\") => return None,
                        _ => {}
                    }
                    let node = self.atom()?;
                    let node = self.scripts(node)?;
                    out.push(node);
                }
                _ => {
                    let node = self.atom()?;
                    let node = self.scripts(node)?;
                    out.push(node);
                }
            }
        }
        Some(merge_numbers(out))
    }

    /// Attaches any `^`, `_` and primes that follow `base`.
    fn scripts(&mut self, base: Node) -> Option<Node> {
        let mut sub = None;
        let mut sup: Option<Node> = None;
        let mut primes = 0;
        // `\sum`, `\lim`, `\max`… take their scripts below and above in display style.
        let mut limits = is_limits_op(&base);
        loop {
            self.skip_space();
            match self.peek() {
                Some('\'') => {
                    self.pos += 1;
                    primes += 1;
                }
                Some('^') if sup.is_none() => {
                    self.pos += 1;
                    sup = Some(self.arg()?);
                }
                Some('_') if sub.is_none() => {
                    self.pos += 1;
                    sub = Some(self.arg()?);
                }
                Some('^' | '_') => return None,
                Some('\\') => match self.peek_command().as_deref() {
                    Some("limits") => {
                        self.command();
                        limits = true;
                    }
                    Some("nolimits") => {
                        self.command();
                        limits = false;
                    }
                    _ => break,
                },
                _ => break,
            }
        }
        if primes > 0 {
            let p = Node::Mo(["′", "″", "‴", "⁗"][(primes - 1).min(3)].into(), Op::Plain);
            sup = Some(match sup {
                Some(s) => Node::Row(vec![p, s]),
                None => p,
            });
        }
        if sub.is_none() && sup.is_none() {
            return Some(base);
        }
        Some(Node::Scripts { base: Box::new(base), sub: sub.map(Box::new), sup: sup.map(Box::new), limits })
    }

    fn char_node(&self, c: char) -> Node {
        if c.is_ascii_digit() {
            return Node::Mn(styled(&c.to_string(), self.font));
        }
        if c.is_alphabetic() {
            let upright = self.font == Some(Font::Upright);
            return Node::Mi(styled(&c.to_string(), self.font), upright);
        }
        let s = match c {
            '-' => "−".to_string(),
            '*' => "∗".to_string(),
            '~' => return Node::Space("0.25em"),
            other => other.to_string(),
        };
        let kind = if matches!(c, '(' | ')' | '[' | ']' | '|') { Op::Fixed } else { Op::Plain };
        Node::Mo(s, kind)
    }

    /// One symbol, command or group. Commands also nest without braces (`\hat\hat…x`), so
    /// every atom counts towards the nesting limit, not only `{…}` groups.
    fn atom(&mut self) -> Option<Node> {
        self.nested(Self::atom_inner)
    }

    fn atom_inner(&mut self) -> Option<Node> {
        self.skip_space();
        let c = self.peek()?;
        match c {
            '{' => {
                self.pos += 1;
                Some(Node::Row(self.expr(Stop::Brace)?))
            }
            '\\' => {
                let name = self.command()?;
                self.cmd(&name)
            }
            '}' | '&' | '^' | '_' => None,
            '#' | '$' | '%' => None,
            _ => {
                self.pos += 1;
                Some(self.char_node(c))
            }
        }
    }

    fn with_font(&mut self, font: Font) -> Option<Node> {
        let saved = self.font;
        self.font = Some(font);
        let n = self.arg();
        self.font = saved;
        n
    }

    fn cmd(&mut self, name: &str) -> Option<Node> {
        if let Some(n) = symbol(name) {
            // `\boldsymbol{\mu}`, `\mathbf{\Sigma}`: style letters given as commands too.
            return Some(match (n, self.font) {
                (Node::Mi(s, upright), Some(f)) => Node::Mi(styled(&s, Some(f)), upright),
                (n, _) => n,
            });
        }
        if let Some(n) = function(name) {
            // TeX puts a thin space after `\log` unless a bracket follows (`\log(x)`).
            self.skip_space();
            let bracket = matches!(self.peek(), Some('(' | '[' | '^' | '_'))
                || matches!(self.peek_command().as_deref(), Some("left" | "big" | "Big" | "bigl" | "Bigl"));
            return Some(match n {
                Node::Row(mut v) if !bracket => {
                    v.push(Node::Space("0.1667em"));
                    Node::Row(v)
                }
                n => n,
            });
        }
        match name {
            // Spacing and escapes.
            "," | "thinspace" => Some(Node::Space("0.1667em")),
            ":" | ">" | "medspace" => Some(Node::Space("0.2222em")),
            ";" | "thickspace" => Some(Node::Space("0.2778em")),
            " " => Some(Node::Space("0.25em")),
            "!" | "negthinspace" => Some(Node::Row(Vec::new())),
            "quad" => Some(Node::Space("1em")),
            "qquad" => Some(Node::Space("2em")),
            "{" | "lbrace" => Some(Node::Mo("{".into(), Op::Fixed)),
            "}" | "rbrace" => Some(Node::Mo("}".into(), Op::Fixed)),
            "%" | "$" | "#" | "&" | "_" => Some(Node::Mo(name.into(), Op::Plain)),
            "|" => Some(Node::Mo("‖".into(), Op::Fixed)),
            "backslash" => Some(Node::Mo("\\".into(), Op::Fixed)),
            // Fractions and roots.
            "frac" | "dfrac" | "tfrac" | "cfrac" => {
                let num = self.arg()?;
                let den = self.arg()?;
                let f = Node::Frac(Box::new(num), Box::new(den), false);
                Some(match name {
                    "dfrac" | "cfrac" => Node::Style(Box::new(f), true),
                    "tfrac" => Node::Style(Box::new(f), false),
                    _ => f,
                })
            }
            "binom" | "dbinom" | "tbinom" => {
                let n = self.arg()?;
                let k = self.arg()?;
                Some(Node::Row(vec![
                    Node::Mo("(".into(), Op::Fence),
                    Node::Frac(Box::new(n), Box::new(k), true),
                    Node::Mo(")".into(), Op::Fence),
                ]))
            }
            "sqrt" => {
                let index = self.optional_bracket()?;
                let body = self.arg()?;
                Some(match index {
                    Some(i) => Node::Root(Box::new(body), Box::new(i)),
                    None => Node::Sqrt(Box::new(body)),
                })
            }
            // Fonts.
            "mathbb" | "Bbb" => self.with_font(Font::DoubleStruck),
            "mathcal" | "mathscr" => self.with_font(Font::Script),
            "mathfrak" => self.with_font(Font::Fraktur),
            "mathbf" => self.with_font(Font::Bold),
            "boldsymbol" | "bm" | "pmb" => self.with_font(Font::BoldItalic),
            "mathit" => self.with_font(Font::Italic),
            "mathrm" | "mathup" => self.with_font(Font::Upright),
            "mathsf" => self.with_font(Font::Sans),
            "mathtt" => self.with_font(Font::Mono),
            "text" | "textrm" | "textnormal" | "mbox" | "textup" | "textsf" | "texttt" => {
                Some(Node::Text(text_of(&self.raw_group()?), false))
            }
            "textbf" => Some(Node::Text(text_of(&self.raw_group()?), true)),
            "textit" | "emph" => Some(Node::Text(text_of(&self.raw_group()?), false)),
            "operatorname" | "operatorname*" | "mathop" => {
                let raw = self.raw_group()?;
                let text = text_of(&raw);
                if name == "operatorname*" {
                    // Takes limits below in display style, like `\max`.
                    return Some(Node::Mo(text.trim().to_string(), Op::Large));
                }
                Some(Node::Row(vec![Node::Mi(text.trim().to_string(), true), Node::Mo("\u{2061}".into(), Op::Plain)]))
            }
            // Accents.
            "hat" => self.accent("^", false),
            "widehat" => self.accent("^", true),
            "check" => self.accent("ˇ", false),
            "tilde" => self.accent("˜", false),
            "widetilde" => self.accent("˜", true),
            "bar" => self.accent("¯", false),
            "overline" => self.accent("¯", true),
            "vec" => self.accent("→", false),
            "overrightarrow" => self.accent("→", true),
            "overleftarrow" => self.accent("←", true),
            "dot" => self.accent("˙", false),
            "ddot" => self.accent("¨", false),
            "acute" => self.accent("´", false),
            "grave" => self.accent("`", false),
            "breve" => self.accent("˘", false),
            "mathring" => self.accent("˚", false),
            "underline" => {
                let base = self.arg()?;
                Some(Node::Under(Box::new(base), Box::new(Node::Mo("_".into(), Op::Stretchy)), true))
            }
            "overbrace" | "underbrace" => {
                let base = self.arg()?;
                let over = name == "overbrace";
                let brace = Node::Mo(if over { "⏞" } else { "⏟" }.into(), Op::Stretchy);
                let mut n = if over {
                    Node::Over(Box::new(base), Box::new(brace), false)
                } else {
                    Node::Under(Box::new(base), Box::new(brace), false)
                };
                // The label goes in the following `^`/`_` and is placed above/below.
                self.skip_space();
                let mark = if over { '^' } else { '_' };
                if self.peek() == Some(mark) {
                    self.pos += 1;
                    let label = self.arg()?;
                    n = if over {
                        Node::Over(Box::new(n), Box::new(label), false)
                    } else {
                        Node::Under(Box::new(n), Box::new(label), false)
                    };
                }
                Some(n)
            }
            "overset" | "stackrel" => {
                let top = self.arg()?;
                let base = self.arg()?;
                Some(Node::Over(Box::new(base), Box::new(top), false))
            }
            "underset" => {
                let bottom = self.arg()?;
                let base = self.arg()?;
                Some(Node::Under(Box::new(base), Box::new(bottom), false))
            }
            // Delimiters.
            "left" => {
                let open = self.delimiter()?;
                let inner = self.expr(Stop::Right)?;
                if self.command().as_deref() != Some("right") {
                    return None;
                }
                let close = self.delimiter()?;
                let mut row = Vec::new();
                if !open.is_empty() {
                    row.push(Node::Mo(open, Op::Fence));
                }
                row.push(Node::Row(inner));
                if !close.is_empty() {
                    row.push(Node::Mo(close, Op::Fence));
                }
                Some(Node::Row(row))
            }
            "middle" => {
                let d = self.delimiter()?;
                Some(Node::Mo(d, Op::Fence))
            }
            "big" | "Big" | "bigg" | "Bigg" | "bigl" | "bigr" | "Bigl" | "Bigr" | "biggl" | "biggr" | "Biggl"
            | "Biggr" | "bigm" | "Bigm" => {
                let d = self.delimiter()?;
                Some(Node::Mo(d, Op::Fixed))
            }
            // Environments.
            "begin" => self.environment(),
            // Styles and things that only change appearance.
            "displaystyle" => Some(Node::Style(Box::new(Node::Row(self.rest_of_group()?)), true)),
            "textstyle" | "scriptstyle" => Some(Node::Style(Box::new(Node::Row(self.rest_of_group()?)), false)),
            "phantom" => Some(Node::Phantom(Box::new(self.arg()?))),
            "boxed" => Some(Node::Boxed(Box::new(self.arg()?))),
            "color" => {
                self.raw_group()?;
                Some(Node::Row(Vec::new()))
            }
            "textcolor" => {
                self.raw_group()?;
                self.arg()
            }
            "label" | "tag" | "hspace" | "vspace" => {
                self.raw_group()?;
                Some(Node::Row(Vec::new()))
            }
            "nonumber" | "notag" | "limits" | "nolimits" | "relax" | "strut" => Some(Node::Row(Vec::new())),
            "not" => {
                // Only the common negated relations.
                let next = self.atom()?;
                let negated = match &next {
                    Node::Mo(s, _) => match s.as_str() {
                        "=" => "≠",
                        "∈" => "∉",
                        "⊂" => "⊄",
                        "⊆" => "⊈",
                        "≡" => "≢",
                        "∼" => "≁",
                        "≤" => "≰",
                        "≥" => "≱",
                        "|" | "∣" => "∤",
                        _ => return None,
                    },
                    _ => return None,
                };
                Some(Node::Mo(negated.into(), Op::Plain))
            }
            "pmod" => {
                let n = self.arg()?;
                Some(Node::Row(vec![
                    Node::Space("0.4444em"),
                    Node::Mo("(".into(), Op::Fixed),
                    Node::Mi("mod".into(), true),
                    Node::Space("0.3333em"),
                    n,
                    Node::Mo(")".into(), Op::Fixed),
                ]))
            }
            "bmod" | "mod" => Some(Node::Mo("mod".into(), Op::Plain)),
            _ => None,
        }
    }

    /// Everything up to the end of the current group (for `\displaystyle`).
    fn rest_of_group(&mut self) -> Option<Vec<Node>> {
        let mut out = Vec::new();
        loop {
            self.skip_space();
            match self.peek() {
                None | Some('}' | '&') => break,
                Some(']') => break,
                Some('\\') if matches!(self.peek_command().as_deref(), Some("right" | "end" | "\\")) => break,
                Some(_) => {
                    let a = self.atom()?;
                    out.push(self.scripts(a)?);
                }
            }
        }
        Some(merge_numbers(out))
    }

    fn accent(&mut self, mark: &str, stretchy: bool) -> Option<Node> {
        let base = self.arg()?;
        let kind = if stretchy { Op::Stretchy } else { Op::Accent };
        Some(Node::Over(Box::new(base), Box::new(Node::Mo(mark.into(), kind)), true))
    }

    /// The delimiter after `\left`, `\right`, `\big`…; `.` means none.
    fn delimiter(&mut self) -> Option<String> {
        self.skip_space();
        let c = self.peek()?;
        if c == '\\' {
            let name = self.command()?;
            let d = match name.as_str() {
                "{" | "lbrace" => "{",
                "}" | "rbrace" => "}",
                "|" | "Vert" | "lVert" | "rVert" => "‖",
                "vert" | "lvert" | "rvert" => "|",
                "langle" => "⟨",
                "rangle" => "⟩",
                "lfloor" => "⌊",
                "rfloor" => "⌋",
                "lceil" => "⌈",
                "rceil" => "⌉",
                "backslash" => "\\",
                "uparrow" => "↑",
                "downarrow" => "↓",
                _ => return None,
            };
            return Some(d.into());
        }
        self.pos += 1;
        match c {
            '.' => Some(String::new()),
            '<' => Some("⟨".into()),
            '>' => Some("⟩".into()),
            '(' | ')' | '[' | ']' | '|' | '/' => Some(c.to_string()),
            _ => None,
        }
    }

    fn environment(&mut self) -> Option<Node> {
        let env = self.raw_group()?;
        let env = env.trim().to_string();
        let (open, close, align) = match env.as_str() {
            "matrix" | "smallmatrix" => ("", "", Align::Center),
            "pmatrix" => ("(", ")", Align::Center),
            "bmatrix" => ("[", "]", Align::Center),
            "Bmatrix" => ("{", "}", Align::Center),
            "vmatrix" => ("|", "|", Align::Center),
            "Vmatrix" => ("‖", "‖", Align::Center),
            "cases" | "dcases" => ("{", "", Align::Left),
            "rcases" => ("", "}", Align::Left),
            "aligned" | "align" | "align*" | "split" | "alignat" | "alignat*" | "flalign" | "flalign*" | "eqnarray"
            | "eqnarray*" => ("", "", Align::Pairs),
            "gathered" | "gather" | "gather*" | "equation" | "equation*" | "multline" | "multline*" => {
                ("", "", Align::Center)
            }
            "array" => {
                // Column specification: ignored apart from being read.
                self.raw_group()?;
                ("", "", Align::Center)
            }
            _ => return None,
        };
        if env.starts_with("alignat") {
            self.raw_group()?;
        }
        let mut rows: Vec<Vec<Node>> = Vec::new();
        let mut row: Vec<Node> = Vec::new();
        loop {
            let cell = self.expr(Stop::Env)?;
            row.push(Node::Row(cell));
            self.skip_space();
            if self.peek() == Some('&') {
                self.pos += 1;
                continue;
            }
            match self.command().as_deref() {
                Some("\\") => {
                    // Optional spacing such as `\\[4pt]`.
                    self.skip_space();
                    if self.peek() == Some('[') {
                        while let Some(c) = self.peek() {
                            self.pos += 1;
                            if c == ']' {
                                break;
                            }
                        }
                    }
                    rows.push(std::mem::take(&mut row));
                }
                Some("end") => {
                    let end = self.raw_group()?;
                    if end.trim() != env {
                        return None;
                    }
                    rows.push(std::mem::take(&mut row));
                    break;
                }
                _ => return None,
            }
        }
        // A trailing `\\` leaves an empty last row.
        if rows.len() > 1 && rows.last().is_some_and(|r| r.iter().all(is_empty)) {
            rows.pop();
        }
        let table = Node::Table { rows, align };
        if open.is_empty() && close.is_empty() {
            return Some(table);
        }
        let mut row = Vec::new();
        if !open.is_empty() {
            row.push(Node::Mo(open.into(), Op::Fence));
        }
        row.push(table);
        if !close.is_empty() {
            row.push(Node::Mo(close.into(), Op::Fence));
        }
        Some(Node::Row(row))
    }
}

fn is_empty(n: &Node) -> bool {
    matches!(n, Node::Row(v) if v.iter().all(is_empty))
}

/// Joins consecutive digits (and a decimal point between digits) into one number.
fn merge_numbers(nodes: Vec<Node>) -> Vec<Node> {
    let mut out: Vec<Node> = Vec::with_capacity(nodes.len());
    let mut i = 0;
    while i < nodes.len() {
        if let Node::Mn(first) = &nodes[i] {
            let mut num = first.clone();
            let mut j = i + 1;
            loop {
                match nodes.get(j) {
                    Some(Node::Mn(d)) => {
                        num.push_str(d);
                        j += 1;
                    }
                    Some(Node::Mo(dot, Op::Plain))
                        if dot == "." && matches!(nodes.get(j + 1), Some(Node::Mn(_))) && !num.contains('.') =>
                    {
                        num.push('.');
                        j += 1;
                    }
                    _ => break,
                }
            }
            out.push(Node::Mn(num));
            i = j;
        } else {
            out.push(nodes[i].clone());
            i += 1;
        }
    }
    out
}

fn is_limits_op(n: &Node) -> bool {
    matches!(n, Node::Mo(s, Op::Large) if !matches!(s.as_str(), "∫" | "∬" | "∭" | "∮"))
}

/// Text inside `\text{…}`: TeX escapes become their characters.
fn text_of(raw: &str) -> String {
    raw.replace('~', "\u{a0}")
}

fn symbol(name: &str) -> Option<Node> {
    let greek = |s: &str| Some(Node::Mi(s.into(), false));
    let upright = |s: &str| Some(Node::Mi(s.into(), true));
    let op = |s: &str| Some(Node::Mo(s.into(), Op::Plain));
    let fixed = |s: &str| Some(Node::Mo(s.into(), Op::Fixed));
    let large = |s: &str| Some(Node::Mo(s.into(), Op::Large));
    match name {
        "alpha" => greek("α"),
        "beta" => greek("β"),
        "gamma" => greek("γ"),
        "delta" => greek("δ"),
        "epsilon" => greek("ϵ"),
        "varepsilon" => greek("ε"),
        "zeta" => greek("ζ"),
        "eta" => greek("η"),
        "theta" => greek("θ"),
        "vartheta" => greek("ϑ"),
        "iota" => greek("ι"),
        "kappa" => greek("κ"),
        "lambda" => greek("λ"),
        "mu" => greek("μ"),
        "nu" => greek("ν"),
        "xi" => greek("ξ"),
        "omicron" => greek("ο"),
        "pi" => greek("π"),
        "varpi" => greek("ϖ"),
        "rho" => greek("ρ"),
        "varrho" => greek("ϱ"),
        "sigma" => greek("σ"),
        "varsigma" => greek("ς"),
        "tau" => greek("τ"),
        "upsilon" => greek("υ"),
        "phi" => greek("ϕ"),
        "varphi" => greek("φ"),
        "chi" => greek("χ"),
        "psi" => greek("ψ"),
        "omega" => greek("ω"),
        "Gamma" => upright("Γ"),
        "Delta" => upright("Δ"),
        "Theta" => upright("Θ"),
        "Lambda" => upright("Λ"),
        "Xi" => upright("Ξ"),
        "Pi" => upright("Π"),
        "Sigma" => upright("Σ"),
        "Upsilon" => upright("Υ"),
        "Phi" => upright("Φ"),
        "Psi" => upright("Ψ"),
        "Omega" => upright("Ω"),
        "infty" => upright("∞"),
        "partial" => upright("∂"),
        "nabla" => upright("∇"),
        "emptyset" | "varnothing" => upright("∅"),
        "hbar" | "hslash" => upright("ℏ"),
        "ell" => upright("ℓ"),
        "Re" => upright("ℜ"),
        "Im" => upright("ℑ"),
        "aleph" => upright("ℵ"),
        "wp" => upright("℘"),
        "top" => upright("⊤"),
        "bot" => upright("⊥"),
        "angle" => upright("∠"),
        "triangle" => upright("△"),
        "imath" => upright("ı"),
        "jmath" => upright("ȷ"),
        "dagger" => op("†"),
        "cdot" => op("⋅"),
        "times" => op("×"),
        "div" => op("÷"),
        "pm" => op("±"),
        "mp" => op("∓"),
        "ast" => op("∗"),
        "star" => op("⋆"),
        "circ" => op("∘"),
        "bullet" => op("∙"),
        "oplus" => op("⊕"),
        "ominus" => op("⊖"),
        "otimes" => op("⊗"),
        "oslash" => op("⊘"),
        "odot" => op("⊙"),
        "wedge" | "land" => op("∧"),
        "vee" | "lor" => op("∨"),
        "cap" => op("∩"),
        "cup" => op("∪"),
        "setminus" => op("∖"),
        "le" | "leq" => op("≤"),
        "ge" | "geq" => op("≥"),
        "leqslant" => op("⩽"),
        "geqslant" => op("⩾"),
        "ne" | "neq" => op("≠"),
        "approx" => op("≈"),
        "sim" => op("∼"),
        "simeq" => op("≃"),
        "cong" => op("≅"),
        "equiv" => op("≡"),
        "propto" => op("∝"),
        "ll" => op("≪"),
        "gg" => op("≫"),
        "prec" => op("≺"),
        "succ" => op("≻"),
        "preceq" => op("⪯"),
        "succeq" => op("⪰"),
        "doteq" => op("≐"),
        "triangleq" => op("≜"),
        "coloneqq" | "coloneq" => op("≔"),
        "in" => op("∈"),
        "notin" => op("∉"),
        "ni" => op("∋"),
        "subset" => op("⊂"),
        "subseteq" => op("⊆"),
        "subsetneq" => op("⊊"),
        "supset" => op("⊃"),
        "supseteq" => op("⊇"),
        "to" | "rightarrow" => op("→"),
        "leftarrow" | "gets" => op("←"),
        "leftrightarrow" => op("↔"),
        "Rightarrow" => op("⇒"),
        "Leftarrow" => op("⇐"),
        "Leftrightarrow" => op("⇔"),
        "iff" => op("⟺"),
        "implies" => op("⟹"),
        "impliedby" => op("⟸"),
        "mapsto" => op("↦"),
        "longrightarrow" => op("⟶"),
        "longleftarrow" => op("⟵"),
        "longmapsto" => op("⟼"),
        "Longrightarrow" => op("⟹"),
        "uparrow" => op("↑"),
        "downarrow" => op("↓"),
        "nearrow" => op("↗"),
        "searrow" => op("↘"),
        "hookrightarrow" => op("↪"),
        "rightleftharpoons" => op("⇌"),
        "mid" => op("∣"),
        "nmid" => op("∤"),
        "parallel" => op("∥"),
        "perp" => op("⊥"),
        "forall" => op("∀"),
        "exists" => op("∃"),
        "nexists" => op("∄"),
        "neg" | "lnot" => op("¬"),
        "colon" => op(":"),
        "vdash" => op("⊢"),
        "models" => op("⊨"),
        "dots" | "ldots" | "dotsc" | "dotso" => op("…"),
        "cdots" | "dotsb" | "dotsm" => op("⋯"),
        "vdots" => op("⋮"),
        "ddots" => op("⋱"),
        "prime" => op("′"),
        "langle" => fixed("⟨"),
        "rangle" => fixed("⟩"),
        "lfloor" => fixed("⌊"),
        "rfloor" => fixed("⌋"),
        "lceil" => fixed("⌈"),
        "rceil" => fixed("⌉"),
        "vert" | "lvert" | "rvert" => fixed("|"),
        "Vert" | "lVert" | "rVert" => fixed("‖"),
        "sum" => large("∑"),
        "prod" => large("∏"),
        "coprod" => large("∐"),
        "bigcup" => large("⋃"),
        "bigcap" => large("⋂"),
        "bigoplus" => large("⨁"),
        "bigotimes" => large("⨂"),
        "bigodot" => large("⨀"),
        "bigvee" => large("⋁"),
        "bigwedge" => large("⋀"),
        "bigsqcup" => large("⨆"),
        "int" => large("∫"),
        "iint" => large("∬"),
        "iiint" => large("∭"),
        "oint" => large("∮"),
        _ => None,
    }
}

/// Function names (`\log`, `\max`, …): upright, followed by an invisible function application.
fn function(name: &str) -> Option<Node> {
    const PLAIN: &[&str] = &[
        "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "coth", "log",
        "ln", "lg", "exp", "det", "dim", "ker", "hom", "deg", "arg", "gcd", "tr", "Tr", "rank", "diag", "sgn",
        "softmax", "erf", "Var", "Cov",
    ];
    const LIMITS: &[(&str, &str)] = &[
        ("lim", "lim"),
        ("limsup", "lim sup"),
        ("liminf", "lim inf"),
        ("max", "max"),
        ("min", "min"),
        ("sup", "sup"),
        ("inf", "inf"),
        ("Pr", "Pr"),
        ("argmax", "arg max"),
        ("argmin", "arg min"),
    ];
    if PLAIN.contains(&name) {
        return Some(Node::Row(vec![Node::Mi(name.into(), true), Node::Mo("\u{2061}".into(), Op::Plain)]));
    }
    LIMITS.iter().find(|(n, _)| *n == name).map(|(_, text)| Node::Mo((*text).into(), Op::Large))
}

/// Maps letters and digits to the Unicode mathematical alphabets (MathML Core renders those,
/// not the old `mathvariant` attribute).
fn styled(s: &str, font: Option<Font>) -> String {
    let Some(font) = font else { return s.to_string() };
    s.chars().map(|c| style_char(c, font)).collect()
}

fn style_char(c: char, font: Font) -> char {
    let exceptions: &[(char, char)] = match font {
        Font::DoubleStruck => &[('C', 'ℂ'), ('H', 'ℍ'), ('N', 'ℕ'), ('P', 'ℙ'), ('Q', 'ℚ'), ('R', 'ℝ'), ('Z', 'ℤ')],
        Font::Script => &[
            ('B', 'ℬ'),
            ('E', 'ℰ'),
            ('F', 'ℱ'),
            ('H', 'ℋ'),
            ('I', 'ℐ'),
            ('L', 'ℒ'),
            ('M', 'ℳ'),
            ('R', 'ℛ'),
            ('e', 'ℯ'),
            ('g', 'ℊ'),
            ('o', 'ℴ'),
        ],
        Font::Fraktur => &[('C', 'ℭ'), ('H', 'ℌ'), ('I', 'ℑ'), ('R', 'ℜ'), ('Z', 'ℨ')],
        Font::Italic => &[('h', 'ℎ')],
        _ => &[],
    };
    if let Some((_, m)) = exceptions.iter().find(|(k, _)| *k == c) {
        return *m;
    }
    let (upper, lower, digit): (u32, u32, Option<u32>) = match font {
        Font::Upright => return c,
        Font::Bold => (0x1D400, 0x1D41A, Some(0x1D7CE)),
        Font::Italic => (0x1D434, 0x1D44E, None),
        Font::BoldItalic => (0x1D468, 0x1D482, Some(0x1D7CE)),
        Font::Script => (0x1D49C, 0x1D4B6, None),
        Font::Fraktur => (0x1D504, 0x1D51E, None),
        Font::DoubleStruck => (0x1D538, 0x1D552, Some(0x1D7D8)),
        Font::Sans => (0x1D5A0, 0x1D5BA, Some(0x1D7E2)),
        Font::Mono => (0x1D670, 0x1D68A, Some(0x1D7F6)),
    };
    let mapped = match c {
        'A'..='Z' => char::from_u32(upper + (c as u32 - 'A' as u32)),
        'a'..='z' => char::from_u32(lower + (c as u32 - 'a' as u32)),
        '0'..='9' => digit.and_then(|d| char::from_u32(d + (c as u32 - '0' as u32))),
        'α'..='ω' if font == Font::BoldItalic => char::from_u32(0x1D736 + (c as u32 - 'α' as u32)),
        'Α'..='Ω' if font == Font::BoldItalic => char::from_u32(0x1D71C + (c as u32 - 'Α' as u32)),
        'α'..='ω' if font == Font::Bold => char::from_u32(0x1D6C2 + (c as u32 - 'α' as u32)),
        'Α'..='Ω' if font == Font::Bold => char::from_u32(0x1D6A8 + (c as u32 - 'Α' as u32)),
        _ => None,
    };
    mapped.unwrap_or(c)
}

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

fn render(n: &Node, out: &mut String) {
    match n {
        Node::Mi(s, upright) => {
            if *upright && s.chars().count() == 1 {
                let _ = write!(out, "<mi mathvariant=\"normal\">{}</mi>", esc(s));
            } else {
                let _ = write!(out, "<mi>{}</mi>", esc(s));
            }
        }
        Node::Mn(s) => {
            let _ = write!(out, "<mn>{}</mn>", esc(s));
        }
        Node::Mo(s, kind) => {
            let attrs = match kind {
                Op::Plain => "",
                // TeX draws `|` and `\|` as ordinary symbols, tight against their
                // neighbours: q(z|x), ‖x‖. Browsers would add operator space around them.
                Op::Fixed if matches!(s.as_str(), "|" | "‖") => " stretchy=\"false\" lspace=\"0\" rspace=\"0\"",
                Op::Fixed => " stretchy=\"false\"",
                Op::Fence => " fence=\"true\" stretchy=\"true\" symmetric=\"true\"",
                Op::Large => " largeop=\"true\" movablelimits=\"true\"",
                Op::Accent => " stretchy=\"false\"",
                Op::Stretchy => " stretchy=\"true\"",
            };
            let _ = write!(out, "<mo{attrs}>{}</mo>", esc(s));
        }
        Node::Text(s, bold) => {
            let style = if *bold { " style=\"font-weight:bold\"" } else { "" };
            // Browsers trim spaces at the edges of `mtext`; `\text{if }` needs its space.
            let lead = s.len() - s.trim_start().len();
            let trail = s.len() - s.trim_end().len();
            let mut t = "\u{a0}".repeat(lead.min(4));
            t.push_str(&esc(s.trim()));
            t.push_str(&"\u{a0}".repeat(trail.min(4)));
            let _ = write!(out, "<mtext{style}>{t}</mtext>");
        }
        Node::Row(items) => {
            if items.len() == 1 {
                render(&items[0], out);
            } else {
                out.push_str("<mrow>");
                for i in items {
                    render(i, out);
                }
                out.push_str("</mrow>");
            }
        }
        Node::Frac(a, b, no_line) => {
            out.push_str(if *no_line { "<mfrac linethickness=\"0\">" } else { "<mfrac>" });
            render(a, out);
            render(b, out);
            out.push_str("</mfrac>");
        }
        Node::Sqrt(a) => {
            out.push_str("<msqrt>");
            render(a, out);
            out.push_str("</msqrt>");
        }
        Node::Root(a, i) => {
            out.push_str("<mroot>");
            render(a, out);
            render(i, out);
            out.push_str("</mroot>");
        }
        Node::Scripts { base, sub, sup, limits } => {
            let (tag, a, b) = match (sub, sup, limits) {
                (Some(s), None, false) => ("msub", Some(s), None),
                (None, Some(s), false) => ("msup", Some(s), None),
                (Some(s), Some(t), false) => ("msubsup", Some(s), Some(t)),
                (Some(s), None, true) => ("munder", Some(s), None),
                (None, Some(s), true) => ("mover", Some(s), None),
                (Some(s), Some(t), true) => ("munderover", Some(s), Some(t)),
                (None, None, _) => {
                    render(base, out);
                    return;
                }
            };
            let _ = write!(out, "<{tag}>");
            render(base, out);
            if let Some(a) = a {
                render(a, out);
            }
            if let Some(b) = b {
                render(b, out);
            }
            let _ = write!(out, "</{tag}>");
        }
        Node::Over(base, over, accent) => {
            out.push_str(if *accent { "<mover accent=\"true\">" } else { "<mover>" });
            render(base, out);
            render(over, out);
            out.push_str("</mover>");
        }
        Node::Under(base, under, accent) => {
            out.push_str(if *accent { "<munder accentunder=\"true\">" } else { "<munder>" });
            render(base, out);
            render(under, out);
            out.push_str("</munder>");
        }
        Node::Table { rows, align } => {
            out.push_str("<mtable>");
            for row in rows {
                out.push_str("<mtr>");
                for (i, cell) in row.iter().enumerate() {
                    let a = match align {
                        Align::Center => "",
                        Align::Left => " style=\"text-align:left\"",
                        Align::Pairs if i % 2 == 0 => " style=\"text-align:right;padding-right:0\"",
                        Align::Pairs => " style=\"text-align:left;padding-left:0\"",
                    };
                    let _ = write!(out, "<mtd{a}>");
                    render(cell, out);
                    out.push_str("</mtd>");
                }
                out.push_str("</mtr>");
            }
            out.push_str("</mtable>");
        }
        Node::Space(w) => {
            let _ = write!(out, "<mspace width=\"{w}\"></mspace>");
        }
        Node::Phantom(a) => {
            out.push_str("<mphantom>");
            render(a, out);
            out.push_str("</mphantom>");
        }
        Node::Boxed(a) => {
            out.push_str("<mrow class=\"math-boxed\">");
            render(a, out);
            out.push_str("</mrow>");
        }
        Node::Style(a, display) => {
            let _ = write!(out, "<mstyle displaystyle=\"{display}\">");
            render(a, out);
            out.push_str("</mstyle>");
        }
    }
}

/// Rewrites `\( … \)` and `\[ … \]` (which Markdown would treat as escaped brackets) to
/// `$ … $` and `$$ … $$`, outside fenced code blocks and code spans. A formula may span
/// lines but not paragraphs.
pub fn normalize_delimiters(src: &str) -> std::borrow::Cow<'_, str> {
    if !src.contains("\\(") && !src.contains("\\[") && !src.contains("$ ") {
        return std::borrow::Cow::Borrowed(src);
    }
    let mut out = String::with_capacity(src.len() + 16);
    let mut prose = String::new();
    let mut in_fence: Option<String> = None;
    for line in src.split_inclusive('\n') {
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();
        let fence: String = trimmed.chars().take_while(|c| *c == '`' || *c == '~').collect();
        let is_fence = fence.chars().count() >= 3
            && indent <= 3
            && (fence.chars().all(|c| c == '`') || fence.chars().all(|c| c == '~'));
        match &in_fence {
            None if is_fence => {
                out.push_str(&rewrite_prose(&prose));
                prose.clear();
                in_fence = Some(fence);
                out.push_str(line);
            }
            None => prose.push_str(line),
            Some(open) => {
                if is_fence && fence.starts_with(open.as_str()) && trimmed.trim_end() == fence {
                    in_fence = None;
                }
                out.push_str(line);
            }
        }
    }
    out.push_str(&rewrite_prose(&prose));
    std::borrow::Cow::Owned(out)
}

fn rewrite_prose(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    // A closing delimiter or code span must be found before the next blank line.
    let mut blank_lines = Vec::new();
    let mut line_start = 0;
    for (j, ch) in chars.iter().enumerate() {
        if *ch == '\n' {
            if chars[line_start..j].iter().all(|c| c.is_whitespace()) {
                blank_lines.push(line_start);
            }
            line_start = j + 1;
        }
    }
    let paragraph_end =
        |from: usize| -> usize { blank_lines.iter().copied().find(|b| *b > from).unwrap_or(chars.len()) };
    // Inside `$…$` (between an opening and a closing dollar on this line).
    let mut dollar_open = false;
    while i < chars.len() {
        let c = chars[i];
        if c == '`' {
            // Copy a code span unchanged (or a lone run of backticks).
            let run = chars[i..].iter().take_while(|c| **c == '`').count();
            let end = paragraph_end(i);
            let mut j = i + run;
            let mut close = None;
            while j < end {
                if chars[j] == '`' {
                    let r = chars[j..].iter().take_while(|c| **c == '`').count();
                    if r == run {
                        close = Some(j + r);
                        break;
                    }
                    j += r;
                } else {
                    j += 1;
                }
            }
            let stop = close.unwrap_or(i + run);
            out.extend(&chars[i..stop]);
            i = stop;
            continue;
        }
        if c == '\n' {
            dollar_open = false;
        }
        // `$$` (display maths) and escaped dollars are left alone and don't pair.
        if c == '$' && chars.get(i + 1) == Some(&'$') {
            out.push_str("$$");
            i += 2;
            continue;
        }
        if c == '$' && i > 0 && chars[i - 1] == '\\' {
            out.push(c);
            i += 1;
            continue;
        }
        // The closing `$` of `$x$` is just copied: only an opening one can start the spaced
        // form below.
        if c == '$' && dollar_open {
            dollar_open = false;
            out.push(c);
            i += 1;
            continue;
        }
        // `$ x^2 $`: Markdown only reads `$x^2$` as maths. Accept the spaced form when it
        // looks like TeX, so prices such as "$ 5 and $ 10" stay text.
        if c == '$' && chars.get(i + 1).is_some_and(|n| *n == ' ' || *n == '\t') {
            let line_end = chars[i + 1..].iter().position(|c| *c == '\n').map_or(chars.len(), |p| i + 1 + p);
            let close = (i + 2..line_end)
                .find(|&j| chars[j] == '$' && chars[j - 1].is_whitespace() && chars.get(j + 1) != Some(&'$'));
            if let Some(j) = close {
                let inner: String = chars[i + 1..j].iter().collect();
                if !inner.trim().is_empty() && inner.contains(['\\', '^', '_', '{', '=']) {
                    out.push('$');
                    out.push_str(inner.trim());
                    out.push('$');
                    i = j + 1;
                    continue;
                }
            }
        }
        if c == '$' {
            dollar_open = true;
            out.push(c);
            i += 1;
            continue;
        }
        if c == '\\' && i + 1 < chars.len() && (chars[i + 1] == '(' || chars[i + 1] == '[') {
            let inline = chars[i + 1] == '(';
            let close = if inline { ')' } else { ']' };
            let end = paragraph_end(i);
            let body_start = i + 2;
            let found = (body_start..end.saturating_sub(1)).find(|&j| chars[j] == '\\' && chars[j + 1] == close);
            if let Some(j) = found {
                let inner: String = chars[body_start..j].iter().collect();
                if inline {
                    out.push('$');
                    out.push_str(inner.trim());
                    out.push('$');
                } else {
                    out.push_str("$$");
                    out.push_str(&inner);
                    out.push_str("$$");
                }
                i = j + 2;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Converts on a 2 MiB stack, the size of the server's worker threads.
    fn on_small_stack(tex: String) -> Option<String> {
        std::thread::Builder::new().stack_size(2 << 20).spawn(move || to_mathml(&tex, false)).unwrap().join().unwrap()
    }

    #[test]
    fn deep_nesting_is_refused_instead_of_overflowing_the_stack() {
        // Commands nest without braces, so these never pass through a `{…}` group. Each
        // input stays just under the 8,000-byte limit.
        for cmd in ["\\hat", "\\not", "\\frac", "\\sqrt", "\\mathbf"] {
            assert_eq!(on_small_stack(format!("{}x", cmd.repeat(7_990 / cmd.len()))), None, "{cmd}");
        }
        assert_eq!(on_small_stack(format!("{}x{}", "{".repeat(3_990), "}".repeat(3_990))), None);
        // Ordinary nesting still works.
        let nested = format!("{}x{}", "\\frac{1}{".repeat(12), "}".repeat(12));
        assert!(on_small_stack(nested).is_some());
        assert!(on_small_stack("\\hat{\\hat{\\vec{x}}}".into()).is_some());
    }

    fn m(tex: &str) -> String {
        to_mathml(tex, false).unwrap_or_else(|| panic!("not converted: {tex}"))
    }

    fn body(tex: &str) -> String {
        let s = m(tex);
        let start = s.find("<semantics>").unwrap() + "<semantics>".len();
        let end = s.find("<annotation").unwrap();
        s[start..end].to_string()
    }

    #[test]
    fn basics() {
        assert_eq!(body("x"), "<mi>x</mi>");
        assert_eq!(body("x^2"), "<msup><mi>x</mi><mn>2</mn></msup>");
        assert_eq!(body("x_{i}^{2}"), "<msubsup><mi>x</mi><mi>i</mi><mn>2</mn></msubsup>");
        assert_eq!(body("3.14"), "<mn>3.14</mn>");
        assert_eq!(body("a-b"), "<mrow><mi>a</mi><mo>−</mo><mi>b</mi></mrow>");
        assert_eq!(body("\\frac{a}{b}"), "<mfrac><mi>a</mi><mi>b</mi></mfrac>");
        assert_eq!(body("\\sqrt{x}"), "<msqrt><mi>x</mi></msqrt>");
        assert_eq!(body("\\sqrt[3]{x}"), "<mroot><mi>x</mi><mn>3</mn></mroot>");
        assert_eq!(body("f'(x)").matches("′").count(), 1);
    }

    #[test]
    fn the_notation_models_write() {
        // Reparameterisation trick, ELBO, attention, diffusion.
        for tex in [
            r"z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)",
            r"\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{\mathrm{KL}}\left(q_\phi(z|x) \,\|\, p(z)\right)",
            r"\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V",
            r"q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t \mathbf{I}\right)",
            r"\nabla_\theta \mathbb{E}_{p(x)}[f_\theta(x)] = \mathbb{E}_{p(x)}[\nabla_\theta f_\theta(x)]",
            r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
            r"\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}",
            r"\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e",
            r"\hat{\theta} = \arg\max_\theta \prod_{i} p(x_i \mid \theta)",
            r"\theta^* = \operatorname*{arg\,max}_{\theta} \log p(\mathcal{D} \mid \theta)",
            r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}",
            r"f(x) = \begin{cases} x^2 & \text{if } x \ge 0 \\ -x & \text{otherwise} \end{cases}",
            r"\begin{aligned} a &= b + c \\ &= d \end{aligned}",
            r"\|x\|_2 = \sqrt{\sum_i x_i^2}",
            r"\mathbb{P}(A \cap B) = \mathbb{P}(A)\,\mathbb{P}(B \mid A)",
            r"\binom{n}{k} = \frac{n!}{k!(n-k)!}",
            r"\vec{v} \cdot \vec{w} = |v||w|\cos\theta",
            r"\overline{x} = \frac{1}{N}\sum_{i=1}^N x_i",
            r"x \in \mathbb{R}^{d}, \; W \in \mathbb{R}^{d \times k}",
            r"\underbrace{a + b}_{\text{sum}}",
        ] {
            assert!(to_mathml(tex, true).is_some(), "not converted: {tex}");
        }
    }

    #[test]
    fn spacing_details() {
        // Thin space after a function name, but not before a bracket.
        assert!(body(r"\log p").contains("<mspace width=\"0.1667em\">"));
        assert!(!body(r"\log(p)").contains("mspace"));
        assert!(body(r"\sin{x}").contains("<mspace width=\"0.1667em\">"));
        // `\,` inside operator names and text is a space, not a comma.
        assert!(body(r"\operatorname*{arg\,max}_x f").contains(">arg max</mo>"));
        // Edge spaces in text survive.
        assert!(body(r"\text{if } x").contains("<mtext>if\u{a0}</mtext>"));
    }

    #[test]
    fn fonts_use_unicode_alphabets() {
        assert!(body(r"\mathbb{E}").contains('𝔼'));
        assert!(body(r"\mathbb{R}").contains('ℝ'));
        assert!(body(r"\mathbb{P}").contains('ℙ'));
        assert!(body(r"\mathcal{N}").contains('𝒩'));
        assert!(body(r"\mathcal{L}").contains('ℒ'));
        assert!(body(r"\mathbf{x}").contains('𝐱'));
        assert!(body(r"\boldsymbol{\mu}").contains('𝝁'));
        assert!(body(r"\mathbf{\Sigma}").contains('𝚺'));
        assert!(body(r"\mathrm{d}").contains("mathvariant=\"normal\""));
    }

    #[test]
    fn limits_and_fences() {
        // Bars are tight like in TeX: q(z|x) and D(q||p), not "q(z | x)" and "q | | p".
        let bar = "<mo stretchy=\"false\" lspace=\"0\" rspace=\"0\">|</mo>";
        assert_eq!(body("q||p").matches(bar).count(), 2, "{}", body("q||p"));
        assert!(body(r"\|x\|").contains("<mo stretchy=\"false\" lspace=\"0\" rspace=\"0\">‖</mo>"));
        let s = body(r"\sum_{i=1}^{n} x_i");
        assert!(s.starts_with("<mrow><munderover><mo largeop=\"true\" movablelimits=\"true\">∑</mo>"), "{s}");
        let s = body(r"\int_0^1 f");
        assert!(s.contains("<msubsup><mo largeop"), "{s}");
        let s = body(r"\left( \frac{a}{b} \right)");
        assert!(s.contains("fence=\"true\" stretchy=\"true\" symmetric=\"true\">(</mo>"), "{s}");
        assert!(body("(a)").contains("<mo stretchy=\"false\">(</mo>"));
        let s = body(r"\left. \frac{df}{dx} \right|_{x=0}");
        assert!(s.contains("<msub>"), "{s}");
    }

    #[test]
    fn unknown_or_broken_input_is_refused() {
        for tex in [
            r"\unknowncommand{x}",
            r"\frac{a}",
            "x^",
            "{a",
            "a}",
            r"\left( a",
            r"\begin{tikzpicture}\end{tikzpicture}",
            r"\begin{matrix} a \end{pmatrix}",
            "x^^2",
            "a & b",
            r"\\",
            "%comment",
        ] {
            assert!(to_mathml(tex, false).is_none(), "should be refused: {tex}");
        }
        assert!(to_mathml(&"x+".repeat(5000), false).is_none());
        let deep = format!("{}x{}", "{".repeat(100), "}".repeat(100));
        assert!(to_mathml(&deep, false).is_none());
    }

    #[test]
    fn output_is_escaped() {
        let s = m(r"a < b \text{<script>alert(1)</script>} > c");
        assert!(!s.contains("<script>"), "{s}");
        assert!(s.contains("&lt;script&gt;"));
        assert!(s.contains("<mo>&lt;</mo>"));
        assert!(s.contains("application/x-tex\">a &lt; b"));
    }

    #[test]
    fn delimiters_are_normalised_outside_code() {
        assert_eq!(normalize_delimiters(r"Inline \( x^2 \) here."), "Inline $x^2$ here.");
        assert_eq!(normalize_delimiters("\\[\nE = mc^2\n\\]\n"), "$$\nE = mc^2\n$$\n");
        assert_eq!(normalize_delimiters(r"Code `\(x\)` stays."), r"Code `\(x\)` stays.");
        assert_eq!(normalize_delimiters("```\n\\(x\\)\n```\n"), "```\n\\(x\\)\n```\n");
        assert_eq!(normalize_delimiters(r"A lone \[ stays"), r"A lone \[ stays");
        assert_eq!(normalize_delimiters("no maths"), "no maths");
        // Spaced dollars, as some models write them.
        assert_eq!(
            normalize_delimiters(r"The derivative of $ x^2 \sin(x) $ is $ \boxed{2x} $."),
            r"The derivative of $x^2 \sin(x)$ is $\boxed{2x}$."
        );
        assert_eq!(normalize_delimiters("It costs $ 5 and $ 10."), "It costs $ 5 and $ 10.");
        assert_eq!(normalize_delimiters("Escaped \\$ x^2 $ stays"), "Escaped \\$ x^2 $ stays");
        // A closing dollar followed by a space is not an opening one (from a real answer).
        let real =
            r"compute $z$ deterministically using the mean ($\mu$) and deviation ($\sigma$) of $q(z|x)$: $ z = \mu $";
        assert_eq!(
            normalize_delimiters(real),
            r"compute $z$ deterministically using the mean ($\mu$) and deviation ($\sigma$) of $q(z|x)$: $z = \mu$"
        );
        assert_eq!(normalize_delimiters("$$ a $$ and $ b^2 $"), "$$ a $$ and $b^2$");
    }
}

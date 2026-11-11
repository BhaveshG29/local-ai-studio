//! How many processor threads the built-in engine uses.
//!
//! Reading a prompt is a few large matrix products, so every core helps. Generating is a chain
//! of small products that all threads must finish before the next token can start, so a few
//! fast cores beat many cores of mixed speed. Measured on an Intel Core Ultra 9 185H (6
//! performance and 10 efficiency cores, 22 threads) with Qwen2.5-0.5B-Instruct Q4_K_M:
//!
//! | | 6 threads (performance cores) | 22 threads (all) | read with 22, write with 6 |
//! |---|---|---|---|
//! | writing, short prompt | 18.8 tokens/s | 6.2–6.4 tokens/s | 18.7 tokens/s |
//! | reading a 1,182-token prompt | ≈ 22 tokens/s | ≈ 41 tokens/s | ≈ 41 tokens/s |
//! | writing after that prompt | 11.9 tokens/s | 4.4 tokens/s | 12.5 tokens/s |
//!
//! So prompts are read on all cores (Rayon's global pool) and tokens are generated on the
//! performance cores (a separate pool). Settings → Inference → *Processor threads*, or
//! `RAYON_NUM_THREADS`, sets one number for both.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadPlan {
    /// Threads for reading prompts (Rayon's global pool).
    pub prompt: usize,
    /// Threads for generating tokens.
    pub generate: usize,
}

impl ThreadPlan {
    /// `fixed`: a number the user chose (applies to both phases). `fast_cores`: the
    /// performance cores (or physical cores on CPUs without efficiency cores).
    pub fn new(fixed: Option<usize>, logical: usize, fast_cores: usize) -> Self {
        let logical = logical.max(1);
        match fixed.filter(|n| *n > 0) {
            Some(n) => Self { prompt: n, generate: n },
            None => Self { prompt: logical, generate: fast_cores.clamp(1, logical) },
        }
    }
}

static PLAN: OnceLock<ThreadPlan> = OnceLock::new();
static GENERATE_POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

fn env_threads() -> Option<usize> {
    std::env::var("RAYON_NUM_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|n| *n > 0)
}

/// Fixes the plan for this process. Call once at start-up, before any model runs, with the
/// number from Settings (if any). Later calls return the plan already chosen.
pub fn configure(setting: Option<usize>) -> ThreadPlan {
    *PLAN.get_or_init(|| {
        let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        // RAYON_NUM_THREADS (set when starting the program) wins over the setting.
        ThreadPlan::new(env_threads().or(setting), logical, crate::system::default_engine_threads())
    })
}

/// The plan in use (the defaults if [`configure`] wasn't called).
pub fn plan() -> ThreadPlan {
    configure(None)
}

/// Runs token generation on the generation threads.
pub fn generating<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let pool = GENERATE_POOL.get_or_init(|| {
        let p = plan();
        if p.generate == p.prompt {
            // The global pool already has this size.
            return None;
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(p.generate)
            .thread_name(|i| format!("las-generate-{i}"))
            .build()
            .map_err(|e| tracing::warn!("could not start the generation threads: {e}"))
            .ok()
    });
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans() {
        // Automatic: all threads read, the fast cores generate.
        assert_eq!(ThreadPlan::new(None, 22, 6), ThreadPlan { prompt: 22, generate: 6 });
        // A chosen number applies to both.
        assert_eq!(ThreadPlan::new(Some(8), 22, 6), ThreadPlan { prompt: 8, generate: 8 });
        // Never more generation threads than the machine has, never zero.
        assert_eq!(ThreadPlan::new(None, 4, 16), ThreadPlan { prompt: 4, generate: 4 });
        assert_eq!(ThreadPlan::new(Some(0), 0, 0), ThreadPlan { prompt: 1, generate: 1 });
    }

    #[test]
    fn generation_runs_on_the_generation_pool() {
        let n = generating(rayon::current_num_threads);
        assert_eq!(n, plan().generate);
    }
}

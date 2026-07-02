use anyhow::Result;
use clap::{Parser, Subcommand};
use std::{env, fs, path::Path};
use vasr_transcribe::{
    EvalArgs, PerfArgs, RunTranscribeArgs, ServeTranscribeArgs, init_logging, run_eval, run_local,
    run_perf, run_transcribe,
};

#[derive(Debug, Parser)]
#[command(
    name = "vasr-transcribe",
    version,
    about = "vASR offline transcribe CLI"
)]
struct Cli {
    /// Show loader, VAD, ASR, and other component runtime logs.
    #[arg(long, short, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the offline transcribe HTTP service.
    Serve(ServeTranscribeArgs),
    /// Transcribe local audio files and write `{stem}-transcription.json` outputs.
    Run(RunTranscribeArgs),
    /// Evaluate service CER with a labeled vASR MessagePack dataset.
    Eval(EvalArgs),
    /// Measure the fastest completion time for an audio source set.
    Perf(PerfArgs),
}

fn load_dotenv() -> Result<()> {
    let path = Path::new(".env");
    if !path.exists() {
        return Ok(());
    }

    let content = fs::read_to_string(path)?;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() || env::var_os(key).is_some() {
            continue;
        }

        let value = value.trim().trim_matches(['\'', '"']);
        unsafe {
            env::set_var(key, value);
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    load_dotenv()?;
    let cli = Cli::parse();
    init_logging(cli.verbose);
    match cli.command {
        Command::Serve(args) => run_transcribe(args).await,
        Command::Run(args) => run_local(args).await,
        Command::Eval(args) => run_eval(args).await,
        Command::Perf(args) => run_perf(args).await,
    }
}

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "vasr-data", version, about = "vASR data utilities")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Convert a FASR AudioList binary file to vASR MessagePack.
    ConvertFasr(ConvertFasrArgs),
}

#[derive(Debug, Clone, Parser)]
struct ConvertFasrArgs {
    /// FASR AudioList binary file.
    #[arg(short = 'i', long, value_name = "PATH")]
    input: PathBuf,

    /// Output vASR `VasrRecordList` MessagePack file.
    #[arg(short = 'o', long, value_name = "PATH")]
    output: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::ConvertFasr(args) => {
            let summary = vasr_data::convert_fasr_audiolist_to_msgpack(&args.input, &args.output)?;
            eprintln!(
                "converted FASR AudioList | records={} input={} output={}",
                summary.records,
                args.input.display(),
                args.output.display()
            );
            Ok(())
        }
    }
}

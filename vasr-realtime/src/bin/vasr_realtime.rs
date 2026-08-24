use anyhow::Result;
use clap::{Parser, Subcommand};
use vasr_realtime::{RealtimeArgs, init_logging, run_realtime};

#[derive(Debug, Parser)]
#[command(
    name = "vasr-realtime",
    version,
    about = "vASR realtime WebSocket service"
)]
struct Cli {
    /// Show VAD, ASR, and other component runtime logs.
    #[arg(long, short, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the realtime WebSocket service.
    Serve(RealtimeArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    match cli.command {
        Command::Serve(args) => run_realtime(args).await,
    }
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::{Cli, Command};

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn serve_subcommand_uses_realtime_defaults() {
        let cli = Cli::try_parse_from(["vasr-realtime", "serve"]).unwrap();
        assert!(matches!(cli.command, Command::Serve(_)));
    }

    #[test]
    fn bare_command_does_not_start_the_service() {
        let error = Cli::try_parse_from(["vasr-realtime"]).unwrap_err();
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
    }

    #[test]
    fn serve_does_not_accept_vad_startup_options() {
        let error =
            Cli::try_parse_from(["vasr-realtime", "serve", "--vad-threshold", "0.5"]).unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
    }
}

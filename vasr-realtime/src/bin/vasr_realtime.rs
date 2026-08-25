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
    /// Start the realtime WebSocket service with command-line model options.
    Serve(RealtimeArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => {
            init_logging(cli.verbose);
            run_realtime(args).await
        }
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
    fn serve_accepts_model_options() {
        let cli = Cli::try_parse_from([
            "vasr-realtime",
            "serve",
            "--model",
            "Qwen/Qwen3-ASR-1.7B",
            "--device",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ])
        .unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve command");
        };
        assert_eq!(args.model.model, "Qwen/Qwen3-ASR-1.7B");
        assert_eq!(args.model.device, "cpu");
        assert_eq!(args.host, "127.0.0.1");
        assert_eq!(args.port, 9000);
    }

    #[test]
    fn init_is_not_a_command() {
        let error = Cli::try_parse_from(["vasr-realtime", "init"]).unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
    }

    #[test]
    fn bare_command_does_not_start_the_service() {
        let error = Cli::try_parse_from(["vasr-realtime"]).unwrap_err();
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
    }
}

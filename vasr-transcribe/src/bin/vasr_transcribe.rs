// 测试里 `expect` 是表达断言失败的常规方式，与 lib 保持一致的豁免。
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::{env, fs, path::Path};
use vasr_transcribe::{
    EvalArgs, PerfArgs, RunTranscribeArgs, ServeTranscribeArgs, StressArgs, init_logging, run_eval,
    run_local, run_perf, run_stress, run_transcribe,
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
    /// Evaluate service CER with a labeled vASR dataset.
    Eval(EvalArgs),
    /// Measure the fastest completion time for an audio source set.
    Perf(PerfArgs),
    /// Continuously stress an HTTP service and sample peak GPU memory.
    Stress(StressArgs),
}

/// 读取当前目录下的 `.env`，只注入进程里还不存在的键。
///
/// 已存在的环境变量优先，CLI 参数又优先于环境变量，因此这里不覆盖已有值。
///
/// # Errors
///
/// `.env` 存在但读取失败时返回错误；文件不存在时直接返回 `Ok(())`。
fn load_dotenv() -> Result<()> {
    let path = Path::new(".env");
    if !path.exists() {
        return Ok(());
    }

    let content = fs::read_to_string(path)?;
    for (key, value) in parse_dotenv_entries(&content) {
        if env::var_os(&key).is_some() {
            continue;
        }
        unsafe {
            env::set_var(key, value);
        }
    }

    Ok(())
}

/// 解析 `.env` 内容：跳过空行和注释，拆掉值两侧成对的引号。
fn parse_dotenv_entries(content: &str) -> Vec<(String, String)> {
    content
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let (key, value) = line.split_once('=')?;
            let key = key.trim();
            if key.is_empty() {
                return None;
            }
            let value = value.trim();
            let value = if value.len() >= 2
                && ((value.starts_with('"') && value.ends_with('"'))
                    || (value.starts_with('\'') && value.ends_with('\'')))
            {
                &value[1..value.len() - 1]
            } else {
                value
            };
            Some((key.to_string(), value.to_string()))
        })
        .collect()
}

/// 加载 `.env`，在 tokio 运行时上执行选中的子命令。
fn main() -> Result<()> {
    load_dotenv()?;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run_cli())
}

/// 解析 CLI 并把子命令分发到对应的实现。
async fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    match cli.command {
        Command::Serve(args) => run_transcribe(args).await,
        Command::Run(args) => run_local(args).await,
        Command::Eval(args) => run_eval(args).await,
        Command::Perf(args) => run_perf(args).await,
        Command::Stress(args) => run_stress(args).await,
    }
}

#[cfg(test)]
mod tests {
    use super::{load_dotenv, parse_dotenv_entries};
    use clap::{Parser, Subcommand};
    use std::{env, fs, path::PathBuf, sync::Mutex, time::SystemTime};
    use vasr_transcribe::ServeTranscribeArgs;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[derive(Debug, Parser)]
    struct EnvTestCli {
        #[command(subcommand)]
        command: EnvTestCommand,
    }

    #[derive(Debug, Subcommand)]
    enum EnvTestCommand {
        Serve(ServeTranscribeArgs),
    }

    fn fresh_test_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        env::temp_dir().join(format!("vasr-dotenv-{}-{nanos}", std::process::id()))
    }

    #[test]
    fn parses_basic_dotenv_assignments_and_ignores_comments() {
        assert_eq!(
            parse_dotenv_entries(
                "# comment\n\n VASR_MODEL = \"local model\" \nVASR_PORT=8867\ninvalid"
            ),
            vec![
                ("VASR_MODEL".to_string(), "local model".to_string()),
                ("VASR_PORT".to_string(), "8867".to_string()),
            ]
        );
    }

    #[test]
    fn loads_dotenv_from_current_directory_and_respects_environment_and_cli_priority() {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let original_dir = env::current_dir().expect("current directory should be readable");
        let test_dir = fresh_test_dir();
        fs::create_dir_all(&test_dir).expect("test directory should be created");

        let max_num_seqs_key = "VASR_MAX_NUM_SEQS";
        let audio_seconds_key = "VASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS";
        let old_max_num_seqs = env::var_os(max_num_seqs_key);
        let old_audio_seconds = env::var_os(audio_seconds_key);
        unsafe {
            env::remove_var(max_num_seqs_key);
            env::remove_var(audio_seconds_key);
        }

        fs::write(
            test_dir.join(".env"),
            "# dotenv values\nVASR_MAX_NUM_SEQS=17\nVASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS='45.0'\n",
        )
        .expect("dotenv file should be written");
        env::set_current_dir(&test_dir).expect("test directory should become current");
        load_dotenv().expect("dotenv should load");

        let parsed = EnvTestCli::try_parse_from(["test", "serve"])
            .expect("dotenv values should enter clap parsing");
        let EnvTestCommand::Serve(args) = parsed.command;
        assert_eq!(args.pipeline.model.max_num_seqs, Some(17));
        assert_eq!(args.pipeline.encoder_batch_max_audio_seconds, Some(45.0));

        let cli_wins = EnvTestCli::try_parse_from([
            "test",
            "serve",
            "--max-num-seqs",
            "9",
            "--encoder-batch-max-audio-seconds",
            "60",
        ])
        .expect("explicit command-line values should parse");
        let EnvTestCommand::Serve(cli_args) = cli_wins.command;
        assert_eq!(cli_args.pipeline.model.max_num_seqs, Some(9));
        assert_eq!(
            cli_args.pipeline.encoder_batch_max_audio_seconds,
            Some(60.0)
        );

        unsafe {
            env::set_var(max_num_seqs_key, "23");
            env::remove_var(audio_seconds_key);
        }
        load_dotenv().expect("dotenv should preserve process environment values");
        let parent_wins = EnvTestCli::try_parse_from(["test", "serve"])
            .expect("process environment should be parsed");
        let EnvTestCommand::Serve(parent_args) = parent_wins.command;
        assert_eq!(parent_args.pipeline.model.max_num_seqs, Some(23));
        assert_eq!(
            parent_args.pipeline.encoder_batch_max_audio_seconds,
            Some(45.0)
        );

        env::set_current_dir(&original_dir).expect("original directory should be restored");
        unsafe {
            env::remove_var(max_num_seqs_key);
            env::remove_var(audio_seconds_key);
            if let Some(value) = old_max_num_seqs {
                env::set_var(max_num_seqs_key, value);
            }
            if let Some(value) = old_audio_seconds {
                env::set_var(audio_seconds_key, value);
            }
        }
        fs::remove_dir_all(test_dir).expect("test directory should be removed");
    }

    #[test]
    fn missing_dotenv_is_ignored() {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let original_dir = env::current_dir().expect("current directory should be readable");
        let test_dir = fresh_test_dir();
        fs::create_dir_all(&test_dir).expect("test directory should be created");
        env::set_current_dir(&test_dir).expect("test directory should become current");
        load_dotenv().expect("missing dotenv should be ignored");
        env::set_current_dir(&original_dir).expect("original directory should be restored");
        fs::remove_dir_all(test_dir).expect("test directory should be removed");
    }
}

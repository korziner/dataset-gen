use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use tracing::info;

// ------------------------------- CLI -------------------------------
#[derive(Parser, Debug)]
#[command(name = "dataset-gen", about = "Generate JSONL dataset using local LLMs", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, clap::Subcommand)]
enum Commands {
    Generate(GenerateArgs),
}

#[derive(Debug, Parser)]
struct GenerateArgs {
    #[arg(long, default_value = "ollama")]
    provider: String,
    #[arg(long)]
    model: String,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    persona: Option<PathBuf>,
    #[arg(long)]
    topic: Option<String>,
    #[arg(long, short = 't', default_value = "4")]
    threads: usize,
    #[arg(long, default_value = "0.7")]
    temperature: f32,
    #[arg(long, default_value = "512")]
    max_tokens: usize,
    #[arg(long)]
    llamacpp_bin: Option<PathBuf>,
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,
    #[arg(long, default_value = "alpaca")]
    format: String,
    #[arg(long)]
    system: Option<String>,
    #[arg(long)]
    sharegpt_id: bool,
    #[arg(long)]
    history: Option<String>,
    /// Default instruction template (use {text} as placeholder)
    #[arg(long, default_value = "Прочитай этотъ отрывокъ и выскажи своё мнѣніе, какъ будто ты обсуждаешь его за обѣдомъ: {text}")]
    instruction_template: String,
}

// ------------------------------- Persona -------------------------------
#[derive(Debug, Deserialize, Serialize)]
struct Persona {
    name: String,
    description: String,
    style: String,
    system_prompt: Option<String>,
}

impl Persona {
    fn from_yaml(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Ok(serde_yaml::from_str(&content)?)
    }

    fn format_prompt(&self, user_text: &str, _topic: &Option<String>) -> String {
        format!(
            "You are {}. {}. Your style: {}. {}\n\nUser: {}\n\nRespond in the style and persona described.",
            self.name, self.description, self.style,
            self.system_prompt.as_ref().map(|s| format!("System: {}", s)).unwrap_or_default(),
            user_text
        )
    }
}

// ------------------------------- LLM Provider Trait -------------------------------
trait LLMProvider: Send + Sync {
    fn generate(&self, prompt: &str) -> Result<String>;
}

// ------------------------------- Ollama Provider -------------------------------
struct OllamaProvider {
    url: String,
    model: String,
    temperature: f32,
    max_tokens: usize,
    client: reqwest::blocking::Client,
}

impl OllamaProvider {
    fn new(url: &str, model: &str, temperature: f32, max_tokens: usize) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            url: url.to_string(),
            model: model.to_string(),
            temperature,
            max_tokens,
            client,
        }
    }
}

impl LLMProvider for OllamaProvider {
    fn generate(&self, prompt: &str) -> Result<String> {
        let preview_prompt = prompt.chars().take(80).collect::<String>();
        eprintln!("[OLLAMA] → {}", preview_prompt.replace('\n', " "));
        let body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        });
        let response = self.client.post(format!("{}/api/generate", self.url))
            .json(&body)
            .send()
            .context("Failed to send request to Ollama")?;
        let status = response.status();
        let text = response.text().unwrap_or_default();
        if !status.is_success() {
            eprintln!("[OLLAMA ERROR] Status: {}, Body: {}", status, text);
            anyhow::bail!("Ollama error {}: {}", status, text);
        }
        let json: serde_json::Value = serde_json::from_str(&text)?;
        let answer = json["response"].as_str().ok_or_else(|| {
            eprintln!("[OLLAMA ERROR] No 'response' field in: {}", text);
            anyhow::anyhow!("No 'response' field in Ollama reply")
        })?;
        let preview_answer = answer.chars().take(100).collect::<String>();
        eprintln!("[OLLAMA] ← {}...", preview_answer);
        Ok(answer.to_string())
    }
}

// ------------------------------- LlamaCpp Provider -------------------------------
struct LlamaCppProvider {
    binary: PathBuf,
    model_path: String,
    temperature: f32,
    max_tokens: usize,
}

impl LlamaCppProvider {
    fn new(bin: PathBuf, model: &str, temperature: f32, max_tokens: usize) -> Self {
        Self {
            binary: bin,
            model_path: model.to_string(),
            temperature,
            max_tokens,
        }
    }
}

impl LLMProvider for LlamaCppProvider {
    fn generate(&self, prompt: &str) -> Result<String> {
        eprintln!("[LLAMACPP] Running {} with model {}", self.binary.display(), self.model_path);
        let output = Command::new(&self.binary)
            .arg("-m")
            .arg(&self.model_path)
            .arg("-p")
            .arg(prompt)
            .arg("--temp")
            .arg(self.temperature.to_string())
            .arg("-n")
            .arg(self.max_tokens.to_string())
            .arg("--no-display-prompt")
            .output()
            .context("Failed to execute llama.cpp")?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("llama.cpp error: {}", stderr);
        }
        let stdout = String::from_utf8(output.stdout)?;
        let answer = stdout.trim();
        let preview = answer.chars().take(100).collect::<String>();
        eprintln!("[LLAMACPP] ← {}...", preview);
        Ok(answer.to_string())
    }
}

// ------------------------------- Форматы вывода -------------------------------
fn parse_history(hist_str: Option<&str>) -> Result<Vec<[String; 2]>> {
    let mut history = Vec::new();
    if let Some(s) = hist_str {
        for pair in s.split('|') {
            let parts: Vec<&str> = pair.splitn(2, ':').collect();
            if parts.len() == 2 {
                history.push([parts[0].to_string(), parts[1].to_string()]);
            } else {
                anyhow::bail!("Invalid history format: {}", pair);
            }
        }
    }
    Ok(history)
}

fn write_example(
    writer: &mut impl Write,
    format: &str,
    instruction: &str,
    input_text: Option<&str>,
    output: &str,
    system: Option<&str>,
    topic: Option<&str>,
    sharegpt_id: bool,
    history: Vec<[String; 2]>,
) -> Result<()> {
    match format {
        "alpaca" => {
            let mut obj = serde_json::Map::new();
            obj.insert("instruction".to_string(), instruction.to_string().into());
            obj.insert("input".to_string(), input_text.unwrap_or("").to_string().into());
            obj.insert("output".to_string(), output.to_string().into());
            if let Some(sys) = system {
                obj.insert("system".to_string(), sys.to_string().into());
            }
            if !history.is_empty() {
                obj.insert("history".to_string(), serde_json::to_value(history)?);
            }
            if let Some(t) = topic {
                obj.insert("topic".to_string(), t.to_string().into());
            }
            writeln!(writer, "{}", serde_json::to_string(&obj)?)?;
        }
        "sharegpt" => {
            let mut conversations = Vec::new();
            if let Some(sys) = system {
                conversations.push(serde_json::json!({"from": "system", "value": sys}));
            }
            for [user, asst] in &history {
                conversations.push(serde_json::json!({"from": "human", "value": user}));
                conversations.push(serde_json::json!({"from": "gpt", "value": asst}));
            }
            let user_msg = if let Some(t) = topic {
                format!("[Topic: {}] {}", t, instruction)
            } else {
                instruction.to_string()
            };
            if let Some(inp) = input_text {
                let full = format!("{}\n\nInput: {}", user_msg, inp);
                conversations.push(serde_json::json!({"from": "human", "value": full}));
            } else {
                conversations.push(serde_json::json!({"from": "human", "value": user_msg}));
            }
            conversations.push(serde_json::json!({"from": "gpt", "value": output}));
            let id = if sharegpt_id { Some(format!("gen_{}", rand::random::<u32>())) } else { None };
            let mut obj = serde_json::Map::new();
            if let Some(id_val) = id {
                obj.insert("id".to_string(), id_val.into());
            }
            obj.insert("conversations".to_string(), conversations.into());
            writeln!(writer, "{}", serde_json::to_string(&obj)?)?;
        }
        "chatml" => {
            let mut messages = Vec::new();
            if let Some(sys) = system {
                messages.push(serde_json::json!({"role": "system", "content": sys}));
            }
            for [user, asst] in &history {
                messages.push(serde_json::json!({"role": "user", "content": user}));
                messages.push(serde_json::json!({"role": "assistant", "content": asst}));
            }
            let user_content = if let Some(t) = topic {
                format!("[Topic: {}] {}", t, instruction)
            } else {
                instruction.to_string()
            };
            let user_full = if let Some(inp) = input_text {
                format!("{}\n\nInput: {}", user_content, inp)
            } else {
                user_content
            };
            messages.push(serde_json::json!({"role": "user", "content": user_full}));
            messages.push(serde_json::json!({"role": "assistant", "content": output}));
            let obj = serde_json::json!({ "messages": messages });
            writeln!(writer, "{}", serde_json::to_string(&obj)?)?;
        }
        _ => anyhow::bail!("Unknown format: {}", format),
    }
    Ok(())
}

// ------------------------------- Main -------------------------------
fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.command {
        Commands::Generate(args) => generate(args),
    }
}

fn generate(args: GenerateArgs) -> Result<()> {
    info!("Loading input from {:?}", args.input);
    let input_file = File::open(&args.input)?;
    let reader = BufReader::new(input_file);
    let mut input_objects: Vec<serde_json::Value> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let mut obj: serde_json::Value = serde_json::from_str(&line)?;
        // Если в строке есть только поле "text", обернём в инструкцию по умолчанию
        if obj.get("instruction").is_none() && obj.get("text").is_some() {
            let text = obj["text"].as_str().unwrap_or("");
            let instruction = args.instruction_template.replace("{text}", text);
            obj["instruction"] = serde_json::Value::String(instruction);
            obj["input"] = serde_json::Value::String("".to_string());
        }
        if obj.get("instruction").is_none() && obj.get("text").is_none() {
            anyhow::bail!("Each input line must have 'text' or 'instruction' field");
        }
        input_objects.push(obj);
    }
    let total = input_objects.len();
    info!("Loaded {} examples", total);

    let persona = if let Some(p) = args.persona {
        Some(Persona::from_yaml(&p)?)
    } else {
        None
    };

    let topics: Vec<Option<String>> = if let Some(topic) = &args.topic {
        vec![Some(topic.clone()); total]
    } else {
        vec![None; total]
    };

    let history = parse_history(args.history.as_deref())?;

    // Инициализация LLM провайдера
    let llm: Arc<dyn LLMProvider> = match args.provider.as_str() {
        "ollama" => {
            info!("Using Ollama with model {} at {}", args.model, args.ollama_url);
            // Проверка соединения
            let test = OllamaProvider::new(&args.ollama_url, &args.model, args.temperature, args.max_tokens);
            if let Err(e) = test.generate("ping") {
                eprintln!("[FATAL] Cannot connect to Ollama: {}. Is Ollama running? Model '{}' exists?", e, args.model);
                std::process::exit(1);
            }
            Arc::new(test)
        }
        "llamacpp" => {
            let bin = args.llamacpp_bin.as_ref().context("--llamacpp-bin required for llamacpp provider")?;
            if !bin.exists() {
                anyhow::bail!("llama.cpp binary not found at {:?}", bin);
            }
            info!("Using llama.cpp binary {:?} with model {}", bin, args.model);
            Arc::new(LlamaCppProvider::new(bin.clone(), &args.model, args.temperature, args.max_tokens))
        }
        other => anyhow::bail!("Unknown provider: {}", other),
    };

    let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build()?;

    let out_file = Arc::new(Mutex::new(File::create(&args.output)?));
    let counter = AtomicUsize::new(0);
    let error_flag = Arc::new(AtomicUsize::new(0));

    let pb = ProgressBar::new(total as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("=> "));

    eprintln!("\n🚀 Starting generation with {} threads...\n", args.threads);

    pool.install(|| {
        input_objects.into_par_iter()
            .zip(topics.into_par_iter())
            .for_each(|(input_obj, topic_opt)| {
                if error_flag.load(Ordering::Relaxed) > 0 {
                    return;
                }
                let idx = counter.fetch_add(1, Ordering::Relaxed) + 1;
                let instruction = input_obj.get("instruction").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let input_text = input_obj.get("input").and_then(|v| v.as_str()).map(|s| s.to_string());
                if instruction.is_empty() {
                    eprintln!("[SKIP] Item {}: empty instruction", idx);
                    pb.inc(1);
                    return;
                }

                // Формируем промпт
                let prompt = if let Some(p) = &persona {
                    let combined = if let Some(inp) = &input_text {
                        format!("{}\n\nInput: {}", instruction, inp)
                    } else {
                        instruction.clone()
                    };
                    p.format_prompt(&combined, &topic_opt)
                } else {
                    if let Some(inp) = &input_text {
                        format!("Instruction: {}\nInput: {}", instruction, inp)
                    } else {
                        instruction.clone()
                    }
                };

                // Генерация ответа
                let output = match llm.generate(&prompt) {
                    Ok(out) => out,
                    Err(e) => {
                        eprintln!("[ERROR] Item {}: {}", idx, e);
                        pb.inc(1);
                        return;
                    }
                };

                // Форматирование записи
                let mut vec_writer = Vec::new();
                if let Err(e) = write_example(
                    &mut vec_writer,
                    &args.format,
                    &instruction,
                    input_text.as_deref(),
                    &output,
                    args.system.as_deref(),
                    topic_opt.as_deref(),
                    args.sharegpt_id,
                    history.clone(),
                ) {
                    eprintln!("[ERROR] Item {}: format error: {}", idx, e);
                    pb.inc(1);
                    return;
                }
                let json_line = String::from_utf8_lossy(&vec_writer).trim_end().to_string();

                // Потоковая запись
                {
                    let mut file = out_file.lock().unwrap();
                    if let Err(e) = writeln!(file, "{}", json_line) {
                        eprintln!("[FATAL] Cannot write to output: {}", e);
                        error_flag.store(1, Ordering::Relaxed);
                        return;
                    }
                    if idx % 50 == 0 {
                        let _ = file.flush();
                    }
                }

                if idx % 100 == 0 {
                    eprintln!("[PROGRESS] {} / {} written", idx, total);
                }
                pb.inc(1);
            });
    });

    pb.finish_with_message("Generation complete");
    info!("Done! Output written to {:?}", args.output);
    Ok(())
}

use axum::{
    extract::State,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use geniex::*;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

#[derive(Deserialize, Debug)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessageDto>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    enable_thinking: bool,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ChatMessageDto {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: ChatMessageDto,
    finish_reason: String,
}

#[derive(Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelData>,
}

#[derive(Serialize)]
struct ModelData {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

struct AppState {
    llm: Mutex<Llm>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== GenieX OpenAI-Compatible API Server ===");

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run --bin openai_server -- <path_to_model.gguf>");
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("[+] Loading model from: {}", model_path);

    init()?;
    println!("[+] SDK Initialized successfully.");

    let config = ModelConfig::default();
    let llm = Llm::create(model_path, "llama_cpp", &config, None, None)?;
    println!("[+] LLM model loaded successfully.");

    let state = Arc::new(AppState {
        llm: Mutex::new(llm),
    });

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    println!("[+] Server listening on http://{}", addr);
    println!("[i] Try endpoints: POST /v1/chat/completions or GET /v1/models");

    axum::serve(listener, app).await.unwrap();

    Ok(())
}

async fn list_models() -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelData {
            id: "local-model".to_string(),
            object: "model".to_string(),
            created: 1677858242,
            owned_by: "geniex-rs".to_string(),
        }],
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> std::result::Result<axum::response::Response, (axum::http::StatusCode, String)> {
    use axum::response::IntoResponse;

    let messages: Vec<ChatMessage> = payload
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();

    if payload.stream {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let state_clone = state.clone();
        let enable_thinking = payload.enable_thinking;

        tokio::spawn(async move {
            let tx_clone = tx.clone();
            tokio::task::spawn_blocking(move || {
                let mut llm = state_clone.llm.blocking_lock();

                let formatted =
                    match llm.apply_chat_template(&messages, None, enable_thinking, true) {
                        Ok(f) => f,
                        Err(e) => {
                            let _ = tx_clone.blocking_send(Err(e));
                            return;
                        }
                    };

                let iter = llm.generate_iter(Some(&formatted), None, None);
                for token_res in iter {
                    if tx_clone.blocking_send(token_res).is_err() {
                        break;
                    }
                }
            })
            .await
            .unwrap();
        });

        let sse_stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(
            |res| -> std::result::Result<Event, Infallible> {
                match res {
                    Ok(token) => {
                        let chunk = serde_json::json!({
                            "id": "chatcmpl-123",
                            "object": "chat.completion.chunk",
                            "created": 1677858242,
                            "model": "local-model",
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": token
                                },
                                "finish_reason": null
                            }]
                        });
                        Ok(Event::default().data(chunk.to_string()))
                    }
                    Err(e) => Ok(Event::default().data(format!("Error: {:?}", e))),
                }
            },
        );

        Ok(Sse::new(sse_stream).into_response())
    } else {
        let mut llm = state.llm.lock().await;

        let formatted = llm
            .apply_chat_template(&messages, None, payload.enable_thinking, true)
            .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("{:?}", e)))?;

        let (full_text, _) = llm
            .generate::<fn(&str) -> bool>(Some(&formatted), None, None, None)
            .map_err(|e| {
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("{:?}", e),
                )
            })?;

        let response = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1677858242,
            model: "local-model".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessageDto {
                    role: "assistant".to_string(),
                    content: full_text,
                },
                finish_reason: "stop".to_string(),
            }],
        };

        Ok(Json(response).into_response())
    }
}

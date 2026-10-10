//! Shared chat-message types and bounded-history helpers for LLM backends.

use serde::Serialize;

/// Default system prompt: encourage short, conversational replies rather than
/// lecture-style answers, since they are spoken aloud.
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are an assistant having a natural spoken conversation. \
Keep replies short, conversational, and to the point. \
Speak like a dialog partner, not a lecture. \
Avoid headings, lists, and long-winded explanations. \
Use one or two sentences unless the user explicitly asks for more detail.";

/// A message in the conversation history (role + content).
#[derive(Serialize, Clone)]
pub struct ChatMessage {
    pub(crate) role: String,
    pub(crate) content: String,
}

impl ChatMessage {
    pub(crate) fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub(crate) fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }

    pub(crate) fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
}

/// Build the messages for a chat request: system prompt (if any) first, then the
/// bounded conversation history plus the new user turn. The system message is
/// inserted only after trimming, so it is never dropped.
pub fn build_request_messages(
    history: &[ChatMessage],
    prompt: &str,
    system: Option<&str>,
    max_turns: usize,
) -> Vec<ChatMessage> {
    let mut messages = history.to_vec();
    messages.push(ChatMessage::user(prompt));
    trim_messages(&mut messages, max_turns);
    if let Some(system) = system {
        messages.insert(0, ChatMessage::system(system));
    }
    messages
}

/// Trim the oldest messages so at most `max_turns` user+assistant pairs remain.
pub fn trim_messages(messages: &mut Vec<ChatMessage>, max_turns: usize) {
    let max_msgs = max_turns * 2;
    if messages.len() > max_msgs {
        let excess = messages.len() - max_msgs;
        messages.drain(0..excess);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_oldest_turns() {
        let mut messages = Vec::new();
        for i in 0..6 {
            messages.push(ChatMessage::user(format!("user {i}")));
            messages.push(ChatMessage::assistant(format!("assistant {i}")));
        }
        trim_messages(&mut messages, 2);
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].content, "user 4");
        assert_eq!(messages[3].content, "assistant 5");
    }
}

import { useEffect, useRef, useState } from "react";
import {
  Heart,
  Send,
  Plus,
  Menu,
  X,
  Sparkles,
  UserRound,
  Bot,
  ShieldCheck,
  MessageCircle,
  ArrowUp,
  LoaderCircle,
  AlertCircle,
} from "lucide-react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

const suggestions = [
  {
    title: "I'm feeling anxious",
    description: "Help me manage my anxiety",
    icon: "🌿",
  },
  {
    title: "I feel overwhelmed",
    description: "There's too much on my mind",
    icon: "☁️",
  },
  {
    title: "I need motivation",
    description: "Help me feel more positive",
    icon: "🌻",
  },
  {
    title: "I just want to talk",
    description: "I'd like someone to listen",
    icon: "💬",
  },
];

function App() {
  const [userId, setUserId] = useState(
    () => localStorage.getItem("aanya_user_id") || ""
  );
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const startConversation = async () => {
    setStarting(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/create_user`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error("Unable to start a conversation. Please try again.");
      }

      const data = await response.json();

      if (!data.user_id) {
        throw new Error("The server did not return a user ID.");
      }

      setUserId(data.user_id);
      localStorage.setItem("aanya_user_id", data.user_id);
      setMessages([]);
      setSidebarOpen(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setStarting(false);
    }
  };

  const sendMessage = async (text = input) => {
    const messageText = text.trim();

    if (!messageText || loading) return;

    setError("");
    setInput("");

    let activeUserId = userId;

    if (!activeUserId) {
      setStarting(true);
      try {
        const createResponse = await fetch(`${API_URL}/create_user`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({}),
        });

        if (!createResponse.ok) {
          throw new Error("Unable to start a conversation.");
        }

        const userData = await createResponse.json();
        activeUserId = userData.user_id;

        if (!activeUserId) {
          throw new Error("The server did not return a user ID.");
        }

        setUserId(activeUserId);
        localStorage.setItem("aanya_user_id", activeUserId);
      } catch (err) {
        setError(err.message || "Unable to connect to the server.");
        setInput(messageText);
        setStarting(false);
        return;
      } finally {
        setStarting(false);
      }
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: messageText,
    };

    setMessages((previous) => [...previous, userMessage]);
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_id: activeUserId,
          message: messageText,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to get a response.");
      }

      if (!data.reply) {
        throw new Error("The server returned an empty response.");
      }

      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.reply,
        },
      ]);
    } catch (err) {
      setError(
        err.message ||
          "I couldn't connect to the server. Please try again."
      );
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const newConversation = () => {
    setUserId("");
    localStorage.removeItem("aanya_user_id");
    setMessages([]);
    setInput("");
    setError("");
    setSidebarOpen(false);
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="sidebar-top">
          <div className="brand">
            <div className="brand-icon">
              <Heart size={21} fill="currentColor" />
            </div>
            <span>Aanya</span>
          </div>

          <button
            className="icon-button mobile-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close menu"
          >
            <X size={20} />
          </button>
        </div>

        <button className="new-chat-button" onClick={newConversation}>
          <Plus size={18} />
          <span>New conversation</span>
        </button>

        <div className="sidebar-section">
          <p className="section-label">YOUR SPACE</p>

          <div className="sidebar-link active">
            <MessageCircle size={17} />
            <span>Current conversation</span>
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="privacy-card">
            <div className="privacy-icon">
              <ShieldCheck size={18} />
            </div>
            <div>
              <p className="privacy-title">Your space, your pace</p>
              <p className="privacy-description">
                Take your time. Share only what feels comfortable.
              </p>
            </div>
          </div>

          <div className="profile">
            <div className="profile-avatar">
              <UserRound size={19} />
            </div>
            <div className="profile-info">
              <span className="profile-name">Your safe space</span>
              <span className="profile-status">Here to listen</span>
            </div>
            <span className="online-dot" />
          </div>
        </div>
      </aside>

      {sidebarOpen && (
        <button
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close sidebar"
        />
      )}

      <main className="main-panel">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-button menu-button"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open menu"
            >
              <Menu size={21} />
            </button>

            <div className="mobile-brand">
              <div className="brand-icon small">
                <Heart size={17} fill="currentColor" />
              </div>
              <span>Aanya</span>
            </div>

            <div className="conversation-heading">
              <span className="conversation-title">Your conversation</span>
              <span className="conversation-subtitle">
                A little space just for you
              </span>
            </div>
          </div>

          <div className="topbar-right">
            <span className="secure-label">
              <ShieldCheck size={15} />
              Private space
            </span>
          </div>
        </header>

        <section className="chat-area">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-orb">
                <div className="orb-glow" />
                <Heart size={34} strokeWidth={1.5} />
                <span className="orb-sparkle sparkle-one">✦</span>
                <span className="orb-sparkle sparkle-two">✧</span>
              </div>

              <div className="welcome-eyebrow">
                <Sparkles size={14} />
                A MOMENT FOR YOU
              </div>

              <h1>
                Hello, I'm <span>Aanya.</span>
              </h1>

              <p className="welcome-description">
                Sometimes, all we need is a space to pause, breathe,
                and talk. I'm here to listen without judgment.
              </p>

              <div className="welcome-prompt">
                <span className="prompt-dot" />
                What's on your mind today?
              </div>

              <div className="suggestion-grid">
                {suggestions.map((item) => (
                  <button
                    className="suggestion-card"
                    key={item.title}
                    onClick={() => sendMessage(item.title)}
                    disabled={loading || starting}
                  >
                    <span className="suggestion-icon">{item.icon}</span>
                    <span className="suggestion-text">
                      <strong>{item.title}</strong>
                      <small>{item.description}</small>
                    </span>
                    <ArrowUp className="suggestion-arrow" size={16} />
                  </button>
                ))}
              </div>

              <p className="welcome-note">
                <ShieldCheck size={14} />
                Start wherever you feel comfortable. There is no
                right or wrong way to feel.
              </p>

              {error && (
                <div className="error-message">
                  <AlertCircle size={17} />
                  {error}
                </div>
              )}

              <button
                className="start-button"
                onClick={startConversation}
                disabled={starting}
              >
                {starting ? (
                  <LoaderCircle className="spin" size={18} />
                ) : (
                  <MessageCircle size={18} />
                )}
                {starting ? "Starting..." : "Start a conversation"}
              </button>
            </div>
          ) : (
            <div className="messages-container">
              <div className="conversation-date">
                <span />
                A space to talk
                <span />
              </div>

              {messages.map((message) => (
                <div
                  className={`message-row ${
                    message.role === "user" ? "user-row" : "assistant-row"
                  }`}
                  key={message.id}
                >
                  {message.role === "assistant" && (
                    <div className="message-avatar assistant-avatar">
                      <Heart size={17} fill="currentColor" />
                    </div>
                  )}

                  <div className="message-content">
                    <span className="message-author">
                      {message.role === "user" ? "You" : "Aanya"}
                    </span>
                    <div
                      className={`message-bubble ${
                        message.role === "user"
                          ? "user-bubble"
                          : "assistant-bubble"
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>

                  {message.role === "user" && (
                    <div className="message-avatar user-avatar">
                      <UserRound size={17} />
                    </div>
                  )}
                </div>
              ))}

              {loading && (
                <div className="message-row assistant-row">
                  <div className="message-avatar assistant-avatar">
                    <Heart size={17} fill="currentColor" />
                  </div>
                  <div className="message-content">
                    <span className="message-author">Aanya</span>
                    <div className="message-bubble assistant-bubble typing-bubble">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </div>
              )}

              {error && (
                <div className="error-message inline-error">
                  <AlertCircle size={17} />
                  {error}
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </section>

        <footer className="composer-area">
          <div className="composer">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Share what's on your mind..."
              rows={1}
              disabled={loading || starting}
              aria-label="Type your message"
            />

            <button
              className="send-button"
              onClick={() => sendMessage()}
              disabled={!input.trim() || loading || starting}
              aria-label="Send message"
            >
              {loading ? (
                <LoaderCircle className="spin" size={19} />
              ) : (
                <Send size={18} />
              )}
            </button>
          </div>

          <div className="composer-disclaimer">
            <span>
              <ShieldCheck size={13} />
              A supportive space, not a substitute for professional care.
            </span>
            <span>Press Enter to send</span>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
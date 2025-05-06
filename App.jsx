import React, { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [messages, setMessages] = useState([]); // { sender, text }
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text) return;             // no empty messages
    setInput("");
    // Add user's message to history
    setMessages((msgs) => [...msgs, { sender: "user", text }]);
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:5000/predict", { question: text });
      const botReply = res.data.response;
      // Add bot's reply
      setMessages((msgs) => [...msgs, { sender: "bot", text: botReply }]);
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        { sender: "bot", text: "Sorry, something went wrong." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-app">
      <h1>Customer Support Chatbot</h1>
      <div className="chat-container">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`message ${m.sender === "user" ? "user" : "bot"}`}
          >
            {m.text}
          </div>
        ))}
        {loading && <div className="typing-indicator">Bot is typing…</div>}
      </div>

      <form className="input-form" onSubmit={sendMessage}>
        <textarea
          rows="2"
          placeholder="Type your question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          {loading ? "Sending…" : "Send"}
        </button>
      </form>
    </div>
  );
}

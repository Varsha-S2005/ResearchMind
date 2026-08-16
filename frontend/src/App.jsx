import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  const uploadPDF = async () => {
    if (!file) {
      setUploadMessage("Please select a PDF first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setUploadMessage("Processing your research paper...");

    try {
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      setUploadMessage(
        `✓ ${data.filename} processed · ${data.chunks_added} chunks indexed`
      );
    } catch (error) {
      setUploadMessage(`Error: ${error.message}`);
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          top_k: 5,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Request failed");
      }

      setAnswer(data.answer);
      setSources(data.sources || []);
    } catch (error) {
      setAnswer(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const exampleQuestion = (text) => {
    setQuestion(text);
  };

  return (
    <div className="app">

      {/* NAVBAR */}
      <nav className="navbar">
        <div className="brand">
          <div className="brand-icon">✦</div>

          <div>
            <div className="brand-name">ResearchMind</div>
            <div className="brand-subtitle">
              Research Intelligence
            </div>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          System Online
        </div>
      </nav>


      {/* HERO */}
      <section className="hero">

        <div className="hero-badge">
          <span>✦</span>
          Retrieval-Augmented Research Assistant
        </div>

        <h1>
          Research smarter.
          <br />
          <span>Discover insights faster.</span>
        </h1>

        <p>
          Upload research papers and ask questions using
          grounded AI answers backed by your sources.
        </p>

      </section>


      <main>


        {/* UPLOAD */}
        <section className="glass-card">

          <div className="section-heading">
            <div className="section-icon">↥</div>

            <div>
              <h2>Upload Research Paper</h2>
              <p>
                Add a PDF to your research knowledge base.
              </p>
            </div>
          </div>


          <div className="upload-box">

            <div className="upload-icon">
              📄
            </div>

            <h3>
              Upload your research paper
            </h3>

            <p>
              PDF files only · Your document will be
              chunked and indexed automatically.
            </p>

            <label className="file-button">
              Choose PDF
              <input
                type="file"
                accept=".pdf"
                onChange={(event) =>
                  setFile(event.target.files[0])
                }
              />
            </label>

            {file && (
              <div className="selected-file">
                ✓ {file.name}
              </div>
            )}

          </div>


          <button
            className="primary-button"
            onClick={uploadPDF}
          >
            <span>Upload & Index</span>
            <span>→</span>
          </button>


          {uploadMessage && (
            <div className="upload-message">
              {uploadMessage}
            </div>
          )}

        </section>


        {/* ASK */}
        <section className="glass-card">

          <div className="section-heading">

            <div className="section-icon">
              ✦
            </div>

            <div>
              <h2>Ask ResearchMind</h2>
              <p>
                Ask questions about your research documents.
              </p>
            </div>

          </div>


          <div className="question-box">

            <textarea
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              placeholder="Ask a research question..."
            />

            <div className="question-footer">

              <span>
                Grounded answers · Source citations
              </span>

              <button
                className="ask-button"
                onClick={askQuestion}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Searching...
                  </>
                ) : (
                  <>
                    Ask ResearchMind →
                  </>
                )}
              </button>

            </div>

          </div>


          <div className="examples">

            <span>Try asking:</span>

            <button
              onClick={() =>
                exampleQuestion(
                  "What are the main challenges of federated learning in vehicular networks?"
                )
              }
            >
              Main challenges
            </button>

            <button
              onClick={() =>
                exampleQuestion(
                  "What security threats affect federated learning?"
                )
              }
            >
              Security threats
            </button>

            <button
              onClick={() =>
                exampleQuestion(
                  "What are the evaluation pitfalls in current research?"
                )
              }
            >
              Evaluation pitfalls
            </button>

          </div>

        </section>


        {/* ANSWER */}
        {answer && (

          <section className="glass-card result-card">

            <div className="result-header">

              <div>
                <div className="result-label">
                  AI RESEARCH RESPONSE
                </div>

                <h2>Answer</h2>
              </div>

              <div className="grounded-badge">
                ✓ Grounded
              </div>

            </div>


            <div className="answer">
              {answer}
            </div>


            {/* SOURCES */}

            <div className="sources-section">

              <div className="result-label">
                SOURCES
              </div>

              <div className="source-grid">

                {sources.length > 0 ? (

                  sources.map((source, index) => (

                    <div
                      className="source-card"
                      key={index}
                    >

                      <div className="source-number">
                        {String(index + 1).padStart(2, "0")}
                      </div>

                      <div>

                        <strong>
                          {source.document_id}
                        </strong>

                        <p>
                          Page {source.page_number}
                          {" · "}
                          Chunk {source.chunk_id}
                        </p>

                      </div>

                    </div>

                  ))

                ) : (

                  <p>No sources available.</p>

                )}

              </div>

            </div>

          </section>

        )}

      </main>


      <footer>
        <span>ResearchMind</span>
        <span>•</span>
        <span>Retrieval-Augmented Generation</span>
        <span>•</span>
        <span>Built with React + FastAPI</span>
      </footer>

    </div>
  );
}

export default App;

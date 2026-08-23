import { useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState("");

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);

  const [verification, setVerification] = useState(null);

  const [loading, setLoading] = useState(false);

  // =====================================================
  // UPLOAD PDF
  // =====================================================

  const uploadPDF = async () => {
    if (!file) {
      setUploadMessage("Please select a PDF first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setUploadMessage("Processing your research paper...");

    try {
      const response = await fetch(
        `${API_URL}/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Upload failed"
        );
      }

      setUploadMessage(
        `✓ ${data.filename} processed · ${data.chunks_added} chunks indexed`
      );
    } catch (error) {
      setUploadMessage(
        `Error: ${error.message}`
      );
    }
  };

  // =====================================================
  // ASK QUESTION
  // =====================================================

  const askQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer("");
    setSources([]);
    setVerification(null);

    try {
      const response = await fetch(
        `${API_URL}/ask`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question,
            top_k: 5,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Request failed"
        );
      }

      setAnswer(data.answer || "");
      setSources(data.sources || []);
      setVerification(
        data.verification || null
      );

    } catch (error) {
      setAnswer(
        `Error: ${error.message}`
      );
      setVerification(null);

    } finally {
      setLoading(false);
    }
  };

  // =====================================================
  // EXAMPLE QUESTIONS
  // =====================================================

  const exampleQuestion = (text) => {
    setQuestion(text);
  };

  // =====================================================
  // VERIFICATION BADGE
  // =====================================================

  const renderVerificationBadge = () => {
    if (!verification) {
      return (
        <div className="grounded-badge">
          ✓ Grounded
        </div>
      );
    }

    if (verification.verdict === "PASS") {
      return (
        <div className="grounded-badge">
          ✓ Grounded
        </div>
      );
    }

    return (
      <div className="grounded-badge">
        ⚠ Verification Failed
      </div>
    );
  };

  return (
    <div className="app">

      {/* =================================================
          NAVBAR
      ================================================= */}

      <nav className="navbar">

        <div className="brand">

          <div className="brand-icon">
            ✦
          </div>

          <div>
            <div className="brand-name">
              ResearchMind
            </div>

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


      {/* =================================================
          HERO
      ================================================= */}

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

        {/* =================================================
            UPLOAD SECTION
        ================================================= */}

        <section className="glass-card">

          <div className="section-heading">

            <div className="section-icon">
              ↥
            </div>

            <div>

              <h2>
                Upload Research Paper
              </h2>

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
                  setFile(
                    event.target.files[0]
                  )
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

            <span>
              Upload & Index
            </span>

            <span>
              →
            </span>

          </button>


          {uploadMessage && (

            <div className="upload-message">
              {uploadMessage}
            </div>

          )}

        </section>


        {/* =================================================
            ASK SECTION
        ================================================= */}

        <section className="glass-card">

          <div className="section-heading">

            <div className="section-icon">
              ✦
            </div>

            <div>

              <h2>
                Ask ResearchMind
              </h2>

              <p>
                Ask questions about your research documents.
              </p>

            </div>

          </div>


          <div className="question-box">

            <textarea
              value={question}
              onChange={(event) =>
                setQuestion(
                  event.target.value
                )
              }
              placeholder="Ask a research question..."
            />


            <div className="question-footer">

              <span>
                Grounded answers · Source citations · AI verification
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


          {/* =================================================
              EXAMPLE QUESTIONS
          ================================================= */}

          <div className="examples">

            <span>
              Try asking:
            </span>


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


        {/* =================================================
            ANSWER SECTION
        ================================================= */}

        {answer && (

          <section className="glass-card result-card">


            {/* RESULT HEADER */}

            <div className="result-header">

              <div>

                <div className="result-label">
                  AI RESEARCH RESPONSE
                </div>

                <h2>
                  Answer
                </h2>

              </div>


              {renderVerificationBadge()}

            </div>


            {/* ANSWER */}

            <div className="answer">
              {answer}
            </div>


            {/* =================================================
                VERIFICATION RESULT
            ================================================= */}

            {verification && (

              <div className="verification-section">

                <div className="result-label">
                  AI VERIFICATION
                </div>


                <div className="verification-card">


                  <div className="verification-header">

                    <strong>
                      Grounding Critic
                    </strong>


                    <span
                      className={
                        verification.verdict === "PASS"
                          ? "verification-pass"
                          : "verification-fail"
                      }
                    >

                      {verification.verdict === "PASS"
                        ? "PASS"
                        : "FAIL"}

                    </span>

                  </div>


                  {/* SCORE */}

                  <div className="verification-score">

                    <span>
                      Grounding Score
                    </span>

                    <strong>
                      {(
                        Number(
                          verification.score || 0
                        ) * 100
                      ).toFixed(0)}
                      %
                    </strong>

                  </div>


                  {/* REASON */}

                  {verification.reason && (

                    <div className="verification-reason">

                      <strong>
                        Critic Reason
                      </strong>

                      <p>
                        {verification.reason}
                      </p>

                    </div>

                  )}


                  {/* UNSUPPORTED CLAIMS */}

                  {verification.unsupported_claims &&
                    verification.unsupported_claims.length > 0 && (

                      <div className="unsupported-claims">

                        <strong>
                          Unsupported Claims
                        </strong>

                        <ul>

                          {verification.unsupported_claims.map(
                            (claim, index) => (

                              <li key={index}>
                                {claim}
                              </li>

                            )
                          )}

                        </ul>

                      </div>

                    )}


                  {/* SUCCESS MESSAGE */}

                  {verification.verdict === "PASS" &&
                    (!verification.unsupported_claims ||
                      verification.unsupported_claims.length === 0) && (

                      <div className="verification-success">

                        ✓ The generated answer is sufficiently
                        supported by the retrieved research evidence.

                      </div>

                    )}

                </div>

              </div>

            )}


            {/* =================================================
                SOURCES
            ================================================= */}

            <div className="sources-section">

              <div className="result-label">
                SOURCES
              </div>


              <div className="source-grid">

                {sources.length > 0 ? (

                  sources.map(
                    (source, index) => (

                      <div
                        className="source-card"
                        key={index}
                      >

                        <div className="source-number">

                          {String(index + 1)
                            .padStart(2, "0")}

                        </div>


                        <div>

                          <strong>
                            {source.document_id}
                          </strong>


                          <p>

                            Page{" "}
                            {source.page_number}

                            {" · "}

                            Chunk{" "}
                            {source.chunk_id}

                          </p>

                        </div>

                      </div>

                    )
                  )

                ) : (

                  <p>
                    No sources available.
                  </p>

                )}

              </div>

            </div>

          </section>

        )}

      </main>


      {/* =================================================
          FOOTER
      ================================================= */}

      <footer>

        <span>
          ResearchMind
        </span>

        <span>
          •
        </span>

        <span>
          Retrieval-Augmented Generation
        </span>

        <span>
          •
        </span>

        <span>
          Built with React + FastAPI
        </span>

      </footer>

    </div>
  );
}

export default App;

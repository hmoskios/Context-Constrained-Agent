# Context-Constrained Codebase Agent

This project implements a context-constrained AI agent for understanding and debugging a C++ codebase under a strict token limit.

This project was built around the [`nlohmann/json`](https://github.com/nlohmann/json) repository, a widely used open-source C++ JSON library with a CMake-based build system and a comprehensive test suite. The agent can answer questions about the codebase, execute build and test commands locally, interpret command outputs, and explain errors.

The core challenge is operating under a **5,000-token limit** while working with a repository that is far too large to fit into context all at once. Instead of loading entire files or the full codebase, the agent selectively retrieves relevant snippets, executes local build/test commands, and constructs a bounded prompt for the model.

The system supports two primary workflows:

### Codebase Understanding
- Locate relevant files and symbols
- Retrieve targeted code snippets
- Explain classes, functions, and architecture

### Build & Debugging
- Execute build and test commands (e.g., `cmake`, `ctest`)
- Capture and parse command output
- Retrieve source context for errors
- Explain failures and test results

---

## Tech Stack

- **Language:** Python
- **LLM:** Google Gemini (`google-genai`)
- **Agent Framework:** LangGraph
- **Code Search:** ripgrep (`rg`)
- **Build System:** CMake, Make, CTest
- **Environment:** Linux / WSL
- **Context Management:** Custom token budgeting + prompt packing

---

## Design Approach

The system is organized into several layers:

### 1. Tooling Layer
The agent uses a set of tools to interact with the codebase:

- `list_dir(...)` - inspects the directory structure
- `read_file(...)` - reads a bounded line range from a file
- `rg_search(...)` - searches the codebase using ripgrep
- `run_cmd(...)` - runs allowed local commands such as `cmake` and `ctest`

### 2. Retrieval Layer
The retriever is responsible for deciding which small pieces of the codebase to load.

For code-understanding queries, it:
- Extracts likely symbols or keywords from the user query
- Searches for likely definitions or matches
- Reads small line windows around those matches
- Ranks and de-duplicates snippets

For build/debug queries, it:
- Parses command output for file/line references
- Reads code windows around those locations
- Returns targeted snippets tied directly to failures

### 3. Context Management Layer
The context manager is the core of the assignment. It:
- Estimates token usage
- Prioritizes prompt contents
- Trims command output
- Selects only the snippets that fit under the total token budget
- Builds a final prompt under the 5,000-token limit

### 4. LangGraph Orchestration Layer
The agent workflow is implemented as a LangGraph state machine. The graph:
- Classifies the query
- Routes to the correct branch
- Retrieves or executes as needed
- Builds the final prompt
- Sends the prompt to Gemini
- Returns the answer

---

## Project Structure

```text
context_agent/
├── agent.py             # Main entry point; invokes the compiled graph
├── main.py              # Command-line interface (CLI)
├── state.py             # Shared data structures (BuildPlan, AgentState)
├── nodes.py             # LangGraph node functions
├── edges.py             # LangGraph routing logic
├── graph.py             # Graph construction and compilation
│
├── tools/
│   ├── __init__.py
│   └── toolkit.py       # list_dir, read_file, rg_search, run_cmd
│
├── retrieval/
│   ├── __init__.py
│   └── retriever.py     # Query retrieval and build-output retrieval
│
├── context/
│   ├── __init__.py
│   ├── budget.py        # Token estimation and context budgeting
│   └── packer.py        # Final prompt assembly
│
├── llm/
│   ├── __init__.py
│   └── gemini_client.py # Gemini API wrapper
│
└── __init__.py
```

---

## Agent Flow

### Understanding Query Flow
1. Receive user query
2. Classify query as a code-understanding query
3. Retrieve relevant snippets from the repository
4. Build a context-limited prompt
5. Send prompt to Gemini
6. Return grounded answer

### Build/Debug Query Flow
1. Receive user query
2. Classify query as a build/debug query
3. Choose a build/test execution plan
4. Run local commands
5. Capture and combine stdout/stderr
6. Parse output for relevant file/line references
7. Retrieve source snippets around those locations
8. Build a context-limited prompt
9. Send prompt to Gemini
10. Return explanation of what happened

---

## LangGraph Workflow

The graph currently follows this structure:

```
START
  ↓
classify_query
  ↓
┌───────────────────────────────┐
│ if understanding query        │
│   → retrieve_understanding    │
│   → build_prompt              │
│   → call_llm                  │
└───────────────────────────────┘
┌───────────────────────────────┐
│ if build/debug query          │
│   → choose_build_plan         │
│   → run_build_plan            │
│   → retrieve_build_snippets   │
│   → build_prompt              │
│   → call_llm                  │
└───────────────────────────────┘
  ↓
 END
```

---

## Context Management Strategy

The repository is intentionally too large to fit into the model context all at once, so the system must actively decide what to include.

### Core strategy

- Never load the whole repository
- Never load full files unless they are already very small
- Retrieve only bounded snippets around relevant matches
- Use the command output to drive retrieval when debugging builds
- Estimate token usage before final prompt assembly
- Drop or trim lower-priority content when needed

### Prioritization

Higher-priority content includes:

- The system prompt
- The user query
- Snippets directly tied to the queried symbol
- Snippets directly tied to build/test failures

Lower-priority content includes:

- Extra background snippets
- Long command output that can be trimmed
- Redundant or overlapping snippets

### Token budgeting

The current implementation uses a simple token estimation strategy: Estimated Tokens ≈ Characters / 4. This estimate is used to:

- Budget prompt contents
- Trim large text blocks
- Decide how many snippets can fit

---

## Setup

### 1. Clone this repository and the target codebase

```bash
git clone https://github.com/hmoskios/Context-Constrained-Agent.git
cd context_agent
git clone https://github.com/nlohmann/json.git
```

The agent expects the repository to already exist locally.

### 2. Instal system dependencies (Linux / WSL)

This project requires several system-level tools:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
sudo apt install ripgrep cmake build-essential git
```

These are required for:

- `cmake` → building the C++ project
- `build-essential` → compiler toolchain
- `ripgrep` → fast code search

### 3. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install langgraph google-genai tiktoken
```

### 5. Set API key

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 6. Run the agent

From the project root, run:

```bash
python3 -m context_agent.main
```

You will be prompted for the path to the local repository, for example:

```bash
/home/your_username/json
```

Then you can ask questions interactively with the agent.

---

## Example Queries
### Codebase Understanding

- `What does json_pointer do and where is it defined?`
- `Which files are responsible for JSON serialization?`
- `What does this file do?`

### Build & Execution

- `Build the project and run the tests. If any test fails, explain why.`
- `Run the tests and report the results.`
- `Why is the build failing? Find the relevant file and explain the error.`

---

## Notes

This project is developed and tested on Linux / WSL. It may require modification to run on Windows without WSL.
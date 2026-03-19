# Context-Constrained Codebase Agent

This project is a context-constrained AI agent for understanding and debugging a real-world C++ codebase under a strict 5,000-token limit.

This project was built around the [`nlohmann/json`](https://github.com/nlohmann/json) repository, a widely used open-source C++ JSON library with a CMake-based build system and a comprehensive test suite. The agent is designed to answer questions about the codebase, execute build and test commands locally, interpret command output, and explain errors - all while carefully controlling how much information is loaded into the model context.

---

## Project Summary

This agent is designed to work with a **pre-downloaded local copy** of a codebase rather than downloading or cloning repositories itself. The core challenge is not simply calling an LLM, but deciding:

- What code or command output is worth loading into context
- How to retrieve only the most relevant sections of a large repository
- How to stay within a strict 5,000-token context budget
- How to combine retrieval, execution, and explanation into a coherent workflow

The system supports two main categories of tasks:

1. **Codebase Understanding**
   - Answer questions about files, classes, functions, and code structure
   - Locate relevant files for a query
   - Explain what a symbol or file does using retrieved snippets

2. **Build & Execution**
   - Run build and test commands locally
   - Capture and interpret command output
   - Parse compiler/test output for file and line references
   - Retrieve the relevant source context for failures
   - Explain what happened and why

---

## Purpose

The purpose of this project is to demonstrate a practical approach to building an AI agent that can work effectively on a codebase that is **too large to fit into the model context all at once**.

Instead of loading the full repository, the agent:

- Uses local tools to inspect and search the codebase
- Retrieves only targeted snippets
- Packs those snippets into a bounded prompt
- Executes local build/test commands when needed
- Uses a LangGraph workflow to coordinate the overall loop

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

## Target Codebase

This project was developed against the following repository: [`nlohmann/json`](https://github.com/nlohmann/json)

This repository was chosen because it is:

- Real-world and non-trivial
- Large enough that it cannot be fully loaded into context
- Built with CMake
- Supported by a strong test suite

The agent assumes the repository already exists locally on disk.

---

## Design Approach

The system is organized into several layers:

### 1. Local Tooling Layer
The agent uses a small set of local tools to interact with the codebase:

- `list_dir(...)` - inspect directory structure
- `read_file(...)` - read a bounded line range from a file
- `rg_search(...)` - search the codebase using ripgrep
- `run_cmd(...)` - run allowed local commands such as `cmake` and `ctest`

This layer gives the agent controlled access to the repository and local execution environment.

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
- Use command output to drive retrieval when debugging builds
- Estimate token usage before final prompt assembly
- Drop or trim lower-priority content when needed

### Prompt contents may include

- System prompt
- User query
- Retrieved code snippets
- Command output
- Short memory summary

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

The current implementation uses a simple token estimation strategy: Estimated tokens ≈ characters / 4

This estimate is used to:

- Budget prompt contents
- Trim large text blocks
- Decide how many snippets can fit

---

## Setup

### 1. Clone this repository and the target codebase

```bash
git clone <your-repo-url>
cd context_agent
git clone https://github.com/nlohmann/json.git
```

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

The agent expects the repository to already exist locally.

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

This project is developed and tested on Linux/WSL. It may require modification to run on Windows without WSL.
# daily-arXiv-astro-ph

This repository was originally forked from [daily-arXiv-ai-enhanced](https://github.com/dw-dengwei/daily-arXiv-ai-enhanced), but it has since been heavily modified and essentially rewritten now. It crawls daily published arXiv articles (focusing on `astro-ph.GA`, `astro-ph.CO`, `astro-ph.IM`), evaluates their relevance, categorizes them, and generates daily summaries using an LLM.

## Repository Structure

To keep the codebase clean and avoid commit conflicts from automated daily updates, this repository uses a split-branch strategy:

- **`main` branch**: Contains all the crawler and summarization code, prompt templates, and local automation scripts.
- **[`data`](https://github.com/Straw674/daily-arXiv-astro-ph/tree/data) branch**: Acts as the storage for the generated daily summaries (Markdown and JSONL files). The automated local runner updates and pushes new data to this branch via a Git Worktree in `dist/` without touching the main codebase.

## Scheduling & Local Execution

The pipeline runs locally on macOS every weekday (Monday to Friday) at 12:43 CST (04:43 UTC) via `launchd` (or cron). Running locally bypasses cloud-runner IP rate limits and firewall blocks on arXiv feeds while ensuring predictable network connectivity.

- **Automated Scheduling**: Configured with a macOS LaunchAgent (`scripts/com.daily-arxiv.astro-ph.plist`) loaded in `~/Library/LaunchAgents/`. If the computer is sleeping or off when the scheduled time arrives, `launchd` automatically executes the task once upon wake.
- **Git Worktree Isolation**: The runner script (`scripts/run_daily.sh`) checks out the `data` branch into `dist/` as a Git Worktree. The working copy of `main` remains untouched, so ongoing code changes never conflict with daily automated updates.
- **Manual Execution & Backfilling**: You can run the pipeline on demand or backfill past dates within arXiv's past week:
  ```bash
  # Run for today
  ./scripts/run_daily.sh

  # Backfill a specific date in the past week
  TARGET_DATE="2026-09-25" ./scripts/run_daily.sh

  # Force regenerate data for a date
  TARGET_DATE="2026-09-25" FORCE_REGEN=true ./scripts/run_daily.sh
  ```

## How It Works

### Paper Relevance & Sorting

To help prioritize which papers to read, this project uses text embeddings and a k-Nearest Neighbors (kNN) approach to calculate personalized relevance:

1. **Reference Library**: The user exports their personal reference library from Zotero as a `.bib` file (which must include paper abstracts).
2. **Embedding Generation**: By running `zotero.py`, this `.bib` file is processed into a `.json` cache containing the embeddings of the reference papers. This serves as a long-term reference for your research interests.
3. **Relevance Calculation**: Each daily arXiv paper's title and abstract are embedded and compared against the reference library using kNN. This generates a **relevance metric** (specifically, the average cosine similarity of the top-k most similar papers in your reference library, where `k` is controlled by `KNN_TOP_K` and defaults to 10) for every daily paper, effectively sorting them according to your personal interests.
4. **Sorting**: Papers are first grouped by topic, and within each topic, they are sorted by this kNN similarity in descending order. Topics themselves are also ordered based on a weighted sum of the similarities of all papers in the group, using an exponential decay (factor of 0.5) according to their position. This balances both the peak relevance and the overall density of interesting papers in each topic.

### Paper Grouping

Papers are categorized into thematic groups to make browsing easier. This can be done in two ways:

- **Manual Grouping (Recommended)**: You can provide a fixed list of group names via the `CUSTOM_GROUPS` environment variable. This ensures consistency and avoids redundant LLM calls.
- **Automatic Grouping**: If `CUSTOM_GROUPS` is not set, the LLM analyzes the titles of the daily papers to dynamically determine appropriate group names based on the content of that specific day.

In both cases, the LLM is responsible for assigning each paper to the most relevant group from the available list.

### Technical Implementation

- **Data Source**: The project uses arXiv's RSS feeds instead of the search API. This ensures the articles fetched are the same batch as the ones on the arXiv website.
- **Filtering**: Only `new` and `cross` submissions are processed. Replacements (updates to old papers) are skipped.

### Daily Summary Format

The output is provided as Markdown files (located in the `data` branch). Each file is structured to give you a quick overview before diving into the details:

- **Table of Contents (ToC)**: At the beginning of the Markdown file, there is a list of links to each paper.
- **Detailed Summaries**: For each paper, the summary is split into two distinct sections:
  - **Background**: Explains the context, the problem domain, and why the research is necessary.
  - **Summary**: Details the specific methods, results, and contributions of the paper.

An example output can be found at [`2026-08-18.md`](2026-08-18.md) in the repository root.

## Setup & Configuration

### 1. Prerequisites & Dependencies
- Python 3.12+ managed by [`uv`](https://docs.astral.sh/uv/)
- Git with write access to the repository

Install dependencies:
```bash
uv sync
```

### 2. Environment Variables (`.env`)
Create a `.env` file in the project root. The pipeline supports any **OpenAI-compatible LLM and text-embedding provider** (e.g. OpenAI, Google Gemini, DeepSeek, DashScope/Qwen, SiliconFlow, or local endpoints via vLLM/Ollama).

The provider credentials listed below (`DASHSCOPE_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`) are merely **examples** showing preset shortcuts that allow switching model families simply by updating `MODEL_NAME`. For general use or other providers, configure `OPENAI_API_KEY` and `OPENAI_BASE_URL`:

```env
# Summarization and Classification Model
MODEL_NAME="gemini-3.1-pro-preview"

# Option A: Generic / Custom OpenAI-Compatible Provider (Works for any provider)
OPENAI_API_KEY="your-api-key"
OPENAI_BASE_URL="https://api.example.com/v1"

# Option B: Pre-configured Provider Shortcuts (Examples)
# DASHSCOPE_API_KEY="your-dashscope-key"
# GEMINI_API_KEY="your-gemini-key"
# DEEPSEEK_API_KEY="your-deepseek-key"

# Embedding API Configuration (Any OpenAI-compatible embedding service)
EMBEDDING_API_KEY="your-embedding-key"
EMBEDDING_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL_NAME="text-embedding-v4"

# Pipeline Parameters
CATEGORIES="astro-ph.GA, astro-ph.CO, astro-ph.IM"
LANGUAGE="中文"
OUTPUT_ROOT="dist"
CONCURRENCY_LIMIT="5"
KNN_TOP_K="10"
```

### 3. Initialize Git Worktree
Initialize the `dist/` directory as a Git Worktree tracking the `data` branch:
```bash
git worktree add dist data
```

### 4. Enable macOS launchd Automation
Install and load the LaunchAgent plist:
```bash
cp scripts/com.daily-arxiv.astro-ph.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.daily-arxiv.astro-ph.plist
```
Execution logs are written to `logs/launchd_stdout.log` and `logs/launchd_stderr.log`.

### 5. Zotero Reference Library Setup
1. Export your personal reference library from Zotero to a `.bib` file (ensure abstracts are included).
2. Save the `.bib` file into `zotero/`.
3. Process the references to generate the embedding cache:
   ```bash
   uv run python src/zotero.py --bib zotero/your_library.bib --output zotero/zotero_embeddings.json
   ```

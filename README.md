# daily-arXiv-astro-ph

This repository crawls daily published arXiv papers in astrophysics (`astro-ph.GA`, `astro-ph.CO`, `astro-ph.IM`), calculates relevance against a reference library using text embeddings, categorizes articles by topic, and generates structured daily summaries using an LLM.

## Repository Structure

The repository maintains an isolated branch structure to separate automation scripts from generated content:

- **`main` branch**: Contains crawler and summarization source code, templates, and local execution scripts.
- **[`data`](https://github.com/Straw674/daily-arXiv-astro-ph/tree/data) branch**: Stores generated daily summaries (Markdown and JSONL files) and the daily summary index. Automated runs commit and push exclusively to this branch via a Git Worktree in `dist/`.

## Scheduling & Execution

The pipeline executes on macOS on weekdays (Monday to Friday) at 20:00 CST (12:00 UTC).

- **Automated Scheduling**: Managed by a launchd agent (`scripts/com.daily-arxiv.astro-ph.plist`) in `~/Library/LaunchAgents/`. If the system is asleep during a scheduled time, launchd triggers execution upon wake.
- **Git Worktree Isolation**: The execution script (`scripts/run_daily.sh`) checks out the `data` branch into `dist/` as a Git Worktree. The working copy on `main` remains untouched, preventing conflicts with local modifications.
- **Manual Execution & Date Override**: The script supports on-demand execution and specific date target overrides:
  ```bash
  # Run for the current date
  ./scripts/run_daily.sh

  # Run for a specific past date
  TARGET_DATE="2026-09-25" ./scripts/run_daily.sh

  # Overwrite existing generated data for a date
  TARGET_DATE="2026-09-25" FORCE_REGEN=true ./scripts/run_daily.sh
  ```

## Methodology

### Relevance Scoring & Ranking

Paper relevance is computed using text embeddings and k-Nearest Neighbors (kNN) comparison against a reference bibliography:

1. **Reference Library**: A Zotero collection exported to `.bib` format containing paper abstracts.
2. **Embedding Cache**: `src/zotero.py` generates vector embeddings for reference abstracts and stores them in a JSON cache.
3. **Similarity Metric**: Titles and abstracts of new papers are embedded and compared against reference vectors. The similarity metric is the mean cosine similarity across the top-$k$ nearest neighbors (`KNN_TOP_K`, default: 10).
4. **Ordering**: Papers within each topic are sorted by descending similarity score. Topics are ordered by an exponentially decayed sum ($\sum \text{score}_i \cdot 0.5^i$) of their constituent paper scores.

### Topic Classification

Papers are classified into thematic categories by the LLM using one of two modes:

- **Predefined Topics**: When `CUSTOM_GROUPS` is set as a comma-separated string, papers are classified into those explicit categories.
- **Dynamic Topics**: When `CUSTOM_GROUPS` is unset, the LLM generates categories based on the batch of daily paper titles.

### Data Ingestion

- **Source**: Daily submissions are retrieved from arXiv RSS feeds.
- **Filter**: Only `new` and `cross` submissions are included. Replacement revisions are excluded.

### Summary Output Format

Daily outputs are stored on the `data` branch with the following structure:

- **Table of Contents**: Anchor links to each paper entry.
- **Paper Entries**:
  - **Background**: Domain context, core concepts, and physical background.
  - **Summary**: Specific methods, results, and findings reported in the paper.

An example output is available at [`2026-08-18.md`](2026-08-18.md).

## Configuration

### 1. Dependencies

Install required dependencies with [`uv`](https://docs.astral.sh/uv/):
```bash
uv sync
```

### 2. Environment Variables (`.env`)

Configure operational parameters and credentials in a root `.env` file. The pipeline supports any OpenAI-compatible LLM and text-embedding API endpoint.

Configure either generic OpenAI-compatible credentials or provider-specific variables:

```env
# Summarization and Classification Model
MODEL_NAME="gemini-3.7-flash"

# Generic OpenAI-Compatible Endpoint
OPENAI_API_KEY="your-api-key"
OPENAI_BASE_URL="https://api.example.com/v1"

# Provider-Specific Presets (Optional)
# DASHSCOPE_API_KEY="your-dashscope-key"
# GEMINI_API_KEY="your-gemini-key"
# DEEPSEEK_API_KEY="your-deepseek-key"

# Embedding API Configuration
EMBEDDING_API_KEY="your-embedding-key"
EMBEDDING_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL_NAME="text-embedding-v4"

# Pipeline Settings
CATEGORIES="astro-ph.GA, astro-ph.CO, astro-ph.IM"
LANGUAGE="中文"
OUTPUT_ROOT="dist"
CONCURRENCY_LIMIT="5"
KNN_TOP_K="10"
```

### 3. Git Worktree Setup

Mount the `data` branch to `dist/`:
```bash
git worktree add dist data
```

### 4. Automated Scheduling Setup

Install and load the launchd agent:
```bash
cp scripts/com.daily-arxiv.astro-ph.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.daily-arxiv.astro-ph.plist
```
Standard output and error streams are directed to `logs/launchd_stdout.log` and `logs/launchd_stderr.log`.

### 5. Reference Library Ingestion

1. Export a reference library from Zotero to `.bib` format with abstracts included.
2. Place the file at `zotero/library.bib`.
3. Generate the embedding cache:
   ```bash
   uv run python src/zotero.py --bib zotero/library.bib --output zotero/zotero_embeddings.json
   ```

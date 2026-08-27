# daily-arXiv-astro-ph

This repository was originally forked from [daily-arXiv-ai-enhanced](https://github.com/dw-dengwei/daily-arXiv-ai-enhanced), but it has since been heavily modified and essentially rewritten now. It crawls daily published arXiv articles (focusing on `astro-ph.GA`, `astro-ph.CO`, `astro-ph.IM`), evaluates their relevance, categorizes them, and generates daily summaries using an LLM.

## Repository Structure

To keep the codebase clean and avoid commit conflicts from automated daily updates, this repository uses a split-branch strategy:

- **`main` branch**: Contains all the crawler and summarization code, prompt templates, and the GitHub Action workflow configurations.
- **[`data`](https://github.com/Straw674/daily-arXiv-astro-ph/tree/data) branch**: Acts as the storage for the generated daily summaries (Markdown and JSONL files). The automation pushes new data to this branch on each run without touching the main codebase.

## Scheduling

The GitHub Actions workflow **does not use a built-in schedule**. Instead, it is triggered externally by [cron-job.org](https://cron-job.org) via a `repository_dispatch` event sent to the GitHub API, running every weekday (Monday to Friday) at 04:43 UTC. This avoids the unreliable delays common with GitHub Actions' native cron scheduler. The workflow can also be triggered manually from the **Actions** tab using the **Run workflow** button, which exposes a `force_regen` option to re-generate data for the current day.

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

## How to Fork and Use

If you want to fork this repository to track your own interests, you will need to complete the following setup steps:

1. **GitHub Secrets & Variables**: In your forked repository, go to `Settings > Secrets and variables > Actions` and configure the following credentials and parameters. The pipeline supports any OpenAI-compatible LLM provider (e.g., DashScope/Qwen, Google Gemini, DeepSeek, OpenAI) for summarization, plus an embedding API.

   **Secrets** (sensitive credentials):

   | Name                 | Description                                              |
   | -------------------- | -------------------------------------------------------- |
   | `DASHSCOPE_API_KEY`  | API key for DashScope / Qwen models                      |
   | `GEMINI_API_KEY`     | API key for Google Gemini models                         |
   | `DEEPSEEK_API_KEY`   | API key for DeepSeek models                              |
   | `OPENAI_API_KEY`     | API key for OpenAI or generic fallback endpoint          |
   | `OPENAI_BASE_URL`    | (Optional) Custom base URL for OpenAI-compatible gateway |
   | `EMBEDDING_API_KEY`  | API key for the text embedding API (e.g. DashScope)      |
   | `EMBEDDING_BASE_URL` | Base URL of the embedding API                            |

   **Variables** (non-sensitive configuration):

   | Name                   | Example                                   | Description                                                        |
   | ---------------------- | ----------------------------------------- | ------------------------------------------------------------------ |
   | `MODEL_NAME`           | `qwen3.8-flash` / `gemini-3.5-flash-lite` | Model name for LLM summarization                                   |
   | `EMBEDDING_MODEL_NAME` | `text-embedding-v4`                       | Model name for text embedding                                      |
   | `CATEGORIES`           | `astro-ph.GA, astro-ph.CO, astro-ph.IM`   | Comma-separated arXiv categories to track                          |
   | `CUSTOM_GROUPS`        | (skipped due to length)                   | Comma-separated list of predefined research topics                 |
   | `LANGUAGE`             | `中文`                                    | Language for the generated summaries                               |
   | `LLM_REASONING_EFFORT` | `high`                                    | (Optional) Reasoning effort for supported models                   |
   | `CONCURRENCY_LIMIT`    | `5`                                       | Number of concurrent LLM calls                                     |
   | `KNN_TOP_K`            | `10`                                      | Number of nearest Zotero papers used for kNN relevance calculation |
   | `NAME`                 | `qx24`                                    | Git committer name for the GitHub Action push                      |
   | `EMAIL`                | `qx24@mails.tsinghua.edu.cn`              | Git committer email for the GitHub Action push                     |

2. **Zotero Library**:
   - Export your personal Zotero library to a `.bib` file. **Make sure to configure the export to include abstracts**.
   - Upload this `.bib` file to the designated directory (`zotero/` by default) in the repository.
3. **Generate Embeddings**:
   - Run the `zotero.py` script locally to process your `.bib` file and generate the `.json` embedding reference file.
   - Commit and push the resulting `.json` file to the repository. This `.json` file will be used by the GitHub Action to evaluate daily papers efficiently, without needing to re-embed your entire library every time.

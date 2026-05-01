# daily-arXiv-astro-ph

This repository was originally forked from [daily-arXiv-ai-enhanced](https://github.com/dw-dengwei/daily-arXiv-ai-enhanced), but it has since been heavily modified and essentially rewritten now. It crawls daily published arXiv articles (focusing on `astro-ph.GA`, `astro-ph.CO`), scores and categorizes them, and generates daily summaries using an LLM.

## Repository Structure

To keep the codebase clean and avoid commit conflicts from automated daily updates, this repository uses a split-branch strategy:

- **`main` branch**: Contains all the crawler and summarization code, prompt templates, and the GitHub Action workflow configurations.
- **`data` branch**: Acts as the storage for the generated daily summaries (Markdown and JSONL files). The GitHub automation pushes new data to this branch daily without touching the main codebase.

## How It Works

### Paper Scoring & Ranking

To help prioritize which papers to read, this project uses text embeddings and a k-Nearest Neighbors (kNN) approach for personalized scoring:

1. **Reference Library**: The user exports their personal reference library from Zotero as a `.bib` file (which must include paper abstracts).
2. **Embedding Generation**: By running `zotero.py`, this `.bib` file is processed into a `.json` cache containing the embeddings of the reference papers. This serves as a long-term reference for your research interests.
3. **Daily Scoring**: Each daily arXiv paper's title and abstract are embedded and compared against the reference library using kNN. This generates a **relevance score** (specifically, the average cosine similarity of the top-k most similar papers in your reference library, where `k` is controlled by `KNN_TOP_K` and defaults to 10) for every daily paper, effectively ranking them according to your personal interests.
4. **Ranking**: Papers are first grouped by topic, and within each topic, they are sorted by this kNN score in descending order. Topics themselves are also ranked based on a weighted sum of the scores of all papers in the group, using an exponential decay (factor of 0.5) according to their rank. This balances both the peak relevance and the overall density of interesting papers in each topic.

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

An example output can be found at [`2026-04-27.md`](2026-04-27.md) in the repository root. For the full archive of daily summaries, see the **[data](https://github.com/Straw674/daily-arXiv-astro-ph/tree/data)** branch.

## How to Fork and Use

If you want to fork this repository to track your own interests, you will need to complete the following setup steps:

1. **GitHub Secrets & Variables**: In your forked repository, go to `Settings > Secrets and variables > Actions` and configure the following (or search `os.getenv` in the codebase for a complete list). At least, You will need two sets of OpenAI-compatible API parameters — one for LLM summarization and one for embedding. Please note that the current implementation in `src/llm.py` uses DeepSeek's specific API calling convention (e.g., `extra_body={"thinking": {"type": "enabled"}}`). If you switch to a different LLM provider, you may need to adjust the parameters in `src/llm.py` to match their specific requirements.

   **Secrets** (sensitive credentials):

   | Name                 | Description                                         |
   | -------------------- | --------------------------------------------------- |
   | `OPENAI_API_KEY`     | API key for the LLM used for summarization          |
   | `OPENAI_BASE_URL`    | Base URL of the OpenAI-compatible summarization API |
   | `EMBEDDING_API_KEY`  | API key for the embedding API                       |
   | `EMBEDDING_BASE_URL` | Base URL of the embedding API                       |

   **Variables** (non-sensitive configuration):

   | Name                   | Example                                 | Description                                          |
   | ---------------------- | --------------------------------------- | ---------------------------------------------------- |
   | `MODEL_NAME`           | `deepseek-v4-pro`                       | Model name for LLM summarization                     |
   | `EMBEDDING_MODEL_NAME` | `text-embedding-v4`                     | Model name for text embedding                        |
   | `CATEGORIES`           | `astro-ph.GA, astro-ph.CO, astro-ph.IM` | Comma-separated arXiv categories to track            |
   | `CUSTOM_GROUPS`        | (skipped due to length)                 | Comma-separated list of predefined research topics   |
   | `LANGUAGE`             | `Chinese`                               | Language for the generated summaries                 |
   | `LLM_REASONING_EFFORT` | `max`                                   | Reasoning effort for the LLM (e.g., max, high)       |
   | `CONCURRENCY_LIMIT`    | `10`                                    | Number of LLM calls to run in parallel               |
   | `KNN_TOP_K`            | `10`                                    | Number of nearest Zotero papers used for kNN scoring |
   | `NAME`                 | `qx24`                                  | Git committer name for the GitHub Action push        |
   | `EMAIL`                | `qx24@mails.tsinghua.edu.cn`            | Git committer email for the GitHub Action push       |

2. **Zotero Library**:
   - Export your personal Zotero library to a `.bib` file. **Make sure to configure the export to include abstracts**.
   - Upload this `.bib` file to the designated directory (`zotero/` by default) in the repository.
3. **Generate Embeddings**:
   - Run the `zotero.py` script locally to process your `.bib` file and generate the `.json` embedding reference file.
   - Commit and push the resulting `.json` file to the repository. This `.json` file will be used by the GitHub Action to score daily papers efficiently, without needing to re-embed your entire library every time.

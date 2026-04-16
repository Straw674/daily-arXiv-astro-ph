# daily-arXiv-astro-ph

- This repository is forked from [daily-arXiv-ai-enhanced](https://github.com/dw-dengwei/daily-arXiv-ai-enhanced), which crawls daily published arXiv articles and summarizes their abstracts using an LLM.
- From 0710 on, the scope has been expanded to cover `astro-ph.GA`, `astro-ph.CO`, and `astro-ph.IM` (previously just `astro-ph.GA`).
- The summaries are generated solely based on the article titles and abstracts, without analyzing the full text of the papers.
- For instructions on how to fork, use, and modify the project, please refer to the original repository.

## Repository Structure

To keep the codebase clean and avoid commit conflicts from automated daily updates, this repository uses a split-branch strategy:

- **`main` branch**: Contains all the crawler and summarization code, prompt templates, and the GitHub Action workflow configurations.
- **`data` branch**: Acts as the storage for the generated daily summaries (Markdown and JSONL files). The GitHub automation pushes new data to this branch daily without touching the main codebase.

## Daily Summaries

The summary files are in the **[data](https://github.com/Straw674/daily-arXiv-astro-ph/tree/data)** branch. 

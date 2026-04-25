import os
from jinja2 import Environment, FileSystemLoader
from typing import List, Dict, Any


def render_daily_markdown(
    papers: List[Dict[str, Any]], date_str: str, output_path: str
):
    env = Environment(loader=FileSystemLoader("src/templates"))
    template = env.get_template("paper.jinja2")

    content = template.render(grouped_data=papers, date=date_str)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def render_readme(data_dir: str, url_prefix: str, output_path: str):
    env = Environment(loader=FileSystemLoader("src/templates"))
    template = env.get_template("readme.jinja2")

    files = [f for f in os.listdir(data_dir) if f.endswith(".md")]
    files.sort(reverse=True)

    dates_and_urls = []
    for f in files:
        date_str = f.replace(".md", "")
        url = os.path.join(url_prefix, f)
        dates_and_urls.append({"date": date_str, "url": url})

    content = template.render(items=dates_and_urls)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

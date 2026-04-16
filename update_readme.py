import argparse
import os
from os.path import join


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory containing generated markdown files")
    parser.add_argument("--output", default="README.md", help="Output README markdown path")
    parser.add_argument("--url-prefix", default="data", help="Prefix used for markdown links")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    template = open("template.md", "r").read()
    data = sorted(os.listdir(args.data_dir), reverse=True)

    readme_content_template = open("readme_content_template.md", "r").read()
    readme_content = "\n\n".join([
        readme_content_template.format(
            date=item.replace(".md", ""), url=join(args.url_prefix, item)
        )
        for item in data
        if item.endswith(".md")
    ])
    markdown = template.format(readme_content=readme_content)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        f.write(markdown)

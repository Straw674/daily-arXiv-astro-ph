today=`date -u "+%Y-%m-%d"`
output_root="${OUTPUT_ROOT:-dist}"
output_data_dir="${output_root}/data"

mkdir -p "${output_data_dir}"

# Task 1: crawl arxiv data
if [ ! -f "${output_data_dir}/${today}.jsonl" ]; then
    echo "Running: scrapy crawl arxiv -o ../${output_data_dir}/${today}.jsonl"
    cd daily_arxiv
    scrapy crawl arxiv -o "../${output_data_dir}/${today}.jsonl"
    cd ..
else
    echo "${output_data_dir}/${today}.jsonl exists, skip"
fi

# Task 2: AI enhance 
if [ ! -f "${output_data_dir}/${today}_AI_enhanced_${LANGUAGE}.jsonl" ]; then
    echo "Running: python enhance.py --data ../${output_data_dir}/${today}.jsonl"
    cd ai
    python enhance.py --data "../${output_data_dir}/${today}.jsonl"
    cd ..
else
    echo "${output_data_dir}/${today}_AI_enhanced_${LANGUAGE}.jsonl exists, skip"
fi

# Task 3: convert
if [ ! -f "${output_data_dir}/${today}.md" ]; then
    echo "Running: python convert.py --data ../${output_data_dir}/${today}_AI_enhanced_${LANGUAGE}.jsonl"
    cd to_md
    python convert.py --data "../${output_data_dir}/${today}_AI_enhanced_${LANGUAGE}.jsonl"
    cd ..
else
    echo "${output_data_dir}/${today}.md exists, skip"
fi

# Update readme
python update_readme.py --data-dir "${output_data_dir}" --output "${output_root}/README.md" --url-prefix "data"
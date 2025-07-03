today=`date -u "+%Y-%m-%d"`

# Task 1: crawl arxiv data
if [ ! -f "data/${today}.jsonl" ]; then
    echo "Running: scrapy crawl arxiv -o ../data/${today}.jsonl"
    cd daily_arxiv
    scrapy crawl arxiv -o ../data/${today}.jsonl
    cd ..
else
    echo "data/${today}.jsonl exists, skip"
fi

# Task 2: AI enhance 
if [ ! -f "data/${today}_AI_enhanced_${LANGUAGE}.jsonl" ]; then
    echo "Running: python enhance.py --data ../data/${today}.jsonl"
    cd ai
    python enhance.py --data ../data/${today}.jsonl
    cd ..
else
    echo "data/${today}_AI_enhanced_${LANGUAGE}.jsonl exists, skip"
fi

# Task 3: convert
if [ ! -f "data/${today}.md" ]; then
    echo "Running: python convert.py --data ../data/${today}_AI_enhanced_${LANGUAGE}.jsonl"
    cd to_md
    python convert.py --data ../data/${today}_AI_enhanced_${LANGUAGE}.jsonl
    cd ..
else
    echo "data/${today}.md exists, skip"
fi

# Update readme
python update_readme.py
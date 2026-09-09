# Hiver AI Support Agent

## Step 1: Setup (5 min)

```bash
git clone <your-repo>
cd hiver-agent

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — add your OPENAI_API_KEY
```

## Step 2: Get the dataset

1. Download from Kaggle: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
2. Place `twcs.csv` in `data/raw/twcs.csv`

## Step 3: Pick your brand

```bash
python scripts/01_eda.py
```

This prints a table of all brands with reply volume, DM-redirect rate, and avg reply length.
It also shows sample conversations. **Pick a brand from the output — note the exact `author_id` string (e.g. `AmazonHelp`).**

## Step 4: Build the dataset

```bash
python scripts/02_build_dataset.py --brand AmazonHelp --max-threads 8000
```

Outputs:
- `data/processed/AmazonHelp_threads.jsonl` — full dataset (JSONL, one thread per line)
- `data/processed/AmazonHelp_sample.jsonl` — 500-thread sample for fast iteration

## Step 5: Discover intents

```bash
python scripts/03_intent_discovery.py --brand AmazonHelp
```

## Step 6: Run the agent

```bash
python scripts/04_run_agent.py --brand AmazonHelp --input "Where is my order #123?"
```

## Step 7: Evaluate

```bash
python scripts/05_evaluate.py --brand AmazonHelp --golden eval/golden_set/AmazonHelp_golden.jsonl
```

---

## Project structure

```
hiver-agent/
├── data/
│   ├── raw/           ← twcs.csv goes here
│   └── processed/     ← generated JSONL datasets
├── eval/
│   └── golden_set/    ← hand-labelled evaluation examples
├── scripts/           ← runnable pipeline scripts (numbered)
├── src/
│   ├── data/          ← load.py, reconstruct.py
│   ├── agent/         ← classifier.py, reply_drafter.py, escalation.py
│   └── eval/          ← harness.py, judge.py
├── requirements.txt
└── .env.example
```

## Thread JSON schema

```json
{
  "thread_id": "123456",
  "brand": "AmazonHelp",
  "messages": [
    {"role": "customer", "text": "...", "tweet_id": "...", "created_at": "..."},
    {"role": "brand",    "text": "...", "tweet_id": "...", "created_at": "..."}
  ],
  "resolved": true,
  "turns": 3
}
```

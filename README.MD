# Nepali embedding model evaluation

This project compares text embedding models on Nepali question-and-answer (Q/A) pairs. The notebook embeds each question and ranks every answer in `qa_pairs.json` by similarity. The answer paired with the question is treated as the correct result.

## Project files

| File or folder | Purpose |
| --- | --- |
| `nepali_rag_embedding_eval.ipynb` | Notebook that loads the Q/A pairs, evaluates models, and displays results. |
| `qa_pairs.json` | Evaluation data: a list of objects with `question` and `answer` fields. |
| `data/section_chunks.json` | Source text chunks. The current notebook does not use these for scoring. |
| `requirements.txt` | Python dependencies for the notebook. |
| `emb_cache/` | Generated question and answer embeddings, cached as NumPy `.npy` files. |
| `embedding_eval_results.csv` | Generated summary of model scores. |

The current `qa_pairs.json` contains 22 pairs. Rows with a missing or empty question or answer are skipped. For each remaining question, the notebook ranks all answers in the file; its paired answer is the single target counted as correct.

## Setup

Use Python 3.13 or a compatible Python version, then create and activate a virtual environment from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Open `nepali_rag_embedding_eval.ipynb` in VS Code or Jupyter and select the `.venv` environment as its Python kernel. Run the cells from top to bottom. The notebook detects CUDA through PyTorch and uses it when available; otherwise, it runs on CPU. Install a PyTorch build compatible with your hardware if you need a specific CUDA setup.

## Models

The notebook includes these models by default:

- `bge-m3`
- `me5-large-instruct`
- `me5-base`
- `qwen3-emb-0.6b`
- `nepali-embedder-v1`
- `minilm-nepali`

Edit `RUN` in the model configuration cell to evaluate only a subset. Model-specific settings such as question prefixes, answer prefixes, prompt names, maximum sequence length, and batch size are defined in `MODELS`.

## Scores and outputs

For each question, the notebook computes similarities between its embedding and every answer embedding. It reports:

- **Recall@1, Recall@3, Recall@5:** the fraction of questions whose paired answer appears within the top 1, 3, or 5 results.
- **MRR (Mean Reciprocal Rank):** the average of `1 / rank` for the paired answer. A rank of 1 contributes 1.0; a rank of 2 contributes 0.5. Higher scores are better.

The results table is sorted by MRR and saved to `embedding_eval_results.csv`. A chart compares Recall@1, Recall@3, and Recall@5. The final cell shows questions where the best-MRR model did not rank the paired answer first.

Embeddings are reused from `emb_cache/` on later runs. Delete that folder to force embeddings to be computed again. The `.gitignore` excludes the cache and generated CSV from version control.

## Evaluation scope

This is a small Q/A answer-retrieval evaluation. Every answer other than the paired answer is treated as incorrect, including an answer that might also be relevant. With only 22 pairs, the scores are useful for an initial comparison but can change noticeably when pairs are added or removed.

The notebook does not currently score retrieval against `data/section_chunks.json`, evaluate generated answers, or measure inference time and memory use. To evaluate chunk retrieval, add human-checked relevant chunk IDs for each question and score the ranked chunks against those labels.
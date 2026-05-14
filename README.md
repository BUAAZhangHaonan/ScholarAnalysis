# ScholarAnalysis

arXiv paper download, Markdown parsing, and focused LLM analysis MCP server.

## Features

- **Paper to Markdown** -- Download arXiv PDFs and convert to clean Markdown via [MinerU](https://github.com/opendatalab/MinerU). Optionally preserve image references for multimodal models.
- **Focused LLM analysis** -- Instead of dumping a full paper into your agent's context, extract only the parts relevant to a specific question using DeepSeek / GLM / Qwen backends with automatic failover.

## Prerequisites

| Dependency | Purpose |
|---|---|
| [ArxivMirror](https://github.com/BUAAZhangHaonan/ArxivMirror) | arXiv PDF mirror and download service |
| [MinerU](https://github.com/opendatalab/MinerU) | PDF to Markdown parsing service |
| Python 3.11+ | Runtime |

## Quick Start

```bash
git clone https://github.com/BUAAZhangHaonan/ScholarAnalysis.git
cd ScholarAnalysis
python -m venv venv && source venv/bin/activate
pip install -e .
```

Create a `.env` file (see `.env.example` for all options):

```env
SCHOLAR_ANALYSIS_ACCESS_TOKEN=your-secret-token
SCHOLAR_ANALYSIS_TRANSPORT=sse
SCHOLAR_ANALYSIS_PORT=8005

SCHOLAR_ANALYSIS_DEEPSEEK_API_KEY=sk-xxx
SCHOLAR_ANALYSIS_ARXIV_MIRROR_BASE_URL=http://127.0.0.1:8900/api/v1
SCHOLAR_ANALYSIS_MINERU_BASE_URL=http://localhost:8888
```

Run the server:

```bash
python -m scholar_analysis.main
```

## MCP Tools

### get_paper_text

Download and parse a paper to Markdown.

```json
{
  "query": "2402.01306",
  "include_images": false
}
```

Returns JSON with paper metadata and full Markdown text.

### analyze_paper

Download, parse, then use LLM to extract content relevant to your question.

```json
{
  "query": "2402.01306",
  "question": "What is the main contribution of this paper?",
  "language": "en",
  "include_images": false
}
```

Returns JSON with a focused analysis result instead of the entire paper.

**Parameters:**

- `query` -- arXiv ID (e.g. `2402.01306`) or arXiv URL. Title search is not supported.
- `question` -- The analysis question or focus area.
- `language` -- `"en"` (default) or `"zh"`.
- `include_images` -- `true` to preserve image references for multimodal models.

## Configuration

All settings use the `SCHOLAR_ANALYSIS_` env prefix and can be placed in a `.env` file. See `.env.example` for the full list.

Key groups:

- **Server** -- `HOST`, `PORT`, `TRANSPORT` (`sse` or `stdio`), `ACCESS_TOKEN`
- **Backends** -- `ARXIV_MIRROR_BASE_URL`, `MINERU_BASE_URL`
- **LLM** -- `DEEPSEEK_*` (primary), `BIGMODEL_*` (GLM fallback), `QWEN_*` (local fallback)

## MCP Client Setup

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "scholar-analysis": {
      "url": "http://localhost:8005/sse",
      "headers": {
        "Authorization": "Bearer your-secret-token"
      }
    }
  }
}
```

## Related Projects

- [ScholarTrace](https://github.com/BUAAZhangHaonan/ScholarTrace) -- Multi-source academic literature search and tracking.

## License

MIT

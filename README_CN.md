# ScholarAnalysis

arXiv 论文下载、Markdown 解析与聚焦式 LLM 分析 MCP 服务器。

## 功能

- **论文转 Markdown** -- 通过 [ArxivMirror](https://github.com/BUAAZhangHaonan/ArxivMirror) 下载 arXiv PDF，经 [MinerU](https://github.com/opendatalab/MinerU) 转为干净的 Markdown。可选择保留图片引用，供多模态模型使用。
- **聚焦式 LLM 分析** -- 不再将整篇论文塞入 Agent 上下文，而是针对具体问题，用 DeepSeek / GLM / Qwen 后端提取相关内容，自动故障转移。

## 前置依赖

| 依赖 | 用途 |
|---|---|
| [ArxivMirror](https://github.com/BUAAZhangHaonan/ArxivMirror) | arXiv PDF 镜像与下载服务 |
| [MinerU](https://github.com/opendatalab/MinerU) | PDF 转 Markdown 解析服务 |
| Python 3.11+ | 运行时 |

## 快速开始

```bash
git clone https://github.com/BUAAZhangHaonan/ScholarAnalysis.git
cd ScholarAnalysis
python -m venv venv && source venv/bin/activate
pip install -e .
```

创建 `.env` 文件（完整选项见 `.env.example`）：

```env
SCHOLAR_ANALYSIS_ACCESS_TOKEN=your-secret-token
SCHOLAR_ANALYSIS_TRANSPORT=sse
SCHOLAR_ANALYSIS_PORT=8005

SCHOLAR_ANALYSIS_DEEPSEEK_API_KEY=sk-xxx
SCHOLAR_ANALYSIS_ARXIV_MIRROR_BASE_URL=http://127.0.0.1:8900/api/v1
SCHOLAR_ANALYSIS_MINERU_BASE_URL=http://localhost:8888
```

启动服务器：

```bash
python -m scholar_analysis.main
```

## MCP 工具

### get_paper_text

下载并解析论文为 Markdown。

```json
{
  "query": "2402.01306",
  "include_images": false
}
```

返回包含论文元数据与完整 Markdown 文本的 JSON。

### analyze_paper

下载、解析后，用 LLM 针对你的问题提取相关内容。

```json
{
  "query": "2402.01306",
  "question": "这篇论文的主要贡献是什么？",
  "language": "zh",
  "include_images": false
}
```

返回聚焦分析结果（而非整篇论文）的 JSON。

**参数说明：**

- `query` -- arXiv ID（如 `2402.01306`）或 arXiv URL，不支持标题搜索。
- `question` -- 分析问题或关注方向。
- `language` -- `"en"`（默认）或 `"zh"`。
- `include_images` -- 设为 `true` 可保留图片引用，供多模态模型使用。

## 配置

所有配置使用 `SCHOLAR_ANALYSIS_` 环境变量前缀，可写入 `.env` 文件。完整列表见 `.env.example`。

主要配置组：

- **服务器** -- `HOST`、`PORT`、`TRANSPORT`（`sse` 或 `stdio`）、`ACCESS_TOKEN`
- **后端服务** -- `ARXIV_MIRROR_BASE_URL`、`MINERU_BASE_URL`
- **LLM** -- `DEEPSEEK_*`（主用）、`BIGMODEL_*`（GLM 备用）、`QWEN_*`（本地备用）

## MCP 客户端配置

在 MCP 客户端中添加：

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

## 相关项目

- [ScholarTrace](https://github.com/BUAAZhangHaonan/ScholarTrace) -- 多源学术文献检索与追踪。

## 许可证

MIT

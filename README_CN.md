# ScholarAnalysis

提供论文取文与可选的聚焦分析 MCP 服务。arXiv 论文使用已有镜像；公开 PDF、
DOI 与出版方页面通过明确的 PDF 链接进入已有 MinerU 解析器。

## 安装与运行

需要 Python 3.11 及以上：

```bash
python -m venv venv
venv/bin/pip install -e .
cp .env.example .env
# 在本地填写凭据和服务地址，不提交 .env。
venv/bin/python -m scholar_analysis.main
```

SSE 地址为 /sse，使用 Authorization Bearer 认证。
systemd 模板位于 scripts/scholar-analysis.service，部署时先核对路径与环境。

## 取文与定位

get_paper_text 从以下三者选一个：query（arXiv 标识、URL 或 DOI）、
pdf_url（PDF 或声明 citation_pdf_url 的出版方页面）、
已有 document_id（只读缓存）。不提供标题搜索，也不绕过付费墙。

```json
{"pdf_url":"https://aclanthology.org/2024.findings-acl.212.pdf","limit_chars":1000}
```

返回 document_id 后，用 offset 和 limit_chars 继续阅读。每次最多 64000 字符。
find_text 从指定 offset 开始定位原文，并返回附近文本。可选 lang 是解析语言提示，
后续分页应保持相同值。refresh=true 配合原始来源显式重新获取和解析。

返回正文、原始/最终来源 URL、文档版本、字符范围、总长度、下一偏移、
是否到末尾、章节标题和行位置。定位基于所选图片模式下 Markdown 的 Unicode 字符，
不能当成 PDF 页码。到达末尾不等于读了全文，还要检查起始位置是否为零。
解析器正文覆盖范围并不认证 PDF 是否完整；解析器返回的表格和公式文本保留。

include_images 只保留图片引用，不获取像素、不代表已看过图像。
有效缓存版本无需访问镜像即可阅读；相同文档并发请求共享解析。
无版本 arXiv 查询可返回已缓存的具体版本，需要最新版本时显式 refresh。

## 可选付费分析

analyze_paper 另接收 question、language（en/zh）、可选的 offset/limit_chars，
以及 analysis_id。可选择方法或附录片段。模型预算不足时只保留完整 Markdown 块，
并声明真正输入的范围；不能因为未读到，就断言全文没有某方法或证明。

相同 analysis_id 和相同输入复用同一任务，包括在途任务、完成结果和失败。
中断但未记下结果的任务不会自动重新收费；新 ID 明确建立新任务。
缓存返回新增费用为零，原始费用另列 original_cost。费用记录不会为了腾空间而自动
删除后重生成；记录存储满时，在发起新模型请求之前报错。主动清空记录会清除幂等历史。

分析只接受正常结束的最终回答，推理内容和达到长度上限的片段都不能冒充成文。
证据引用逐字匹配实际输入，返回原文位置；这仅检查出处，不认证逻辑支持关系。
缺少材料和输入截断明确保留；文本分析不声称读过图片。
结果未知的已发请求不自动重试；有限的 HTTP 429 重试只使用当前模型。

默认模型为 deepseek-flash；需要 Pro 时显式配置 deepseek-v4-pro。
不自动切换供应商或模型。thinking=false/true 分别明确发送 disabled/enabled。
账号并发上限不是服务默认工作线程数；PDF、HTTP 和模型并发分别按实际能力配置。

## 费用、错误与缓存

所有公共工具成功、操作失败和缓存读取均返回 mcp.cost.v1：
CNY 使用量估算、逐次请求、未知费用与上下界，不是供应商账单。
范围只包含供应商模型 API，解析、服务器和 GPU 成本不在其中。
跨峰谷时段或缺少缓存命中明细时保留区间；缺失用量绝不算作零。

错误提供 error_code、stage、retryable、retry_after_seconds。
例如 mirror_download 阶段 HTTP 502 属于镜像下载服务，不是 MinerU 故障。
排队和整体请求都有时限，原始诊断留在服务日志。

缓存只接受有效正文并原子写入，来源目录不使用内容哈希。
分析记录含问题、结果和费用，应限制缓存目录访问。YAML 模板随安装包分发，
不依赖仓库工作目录；可通过 prompts_dir 指定自己的模板。

## 验证与压测

```bash
# 离线，不访问网络、不调用模型
venv/bin/python -m unittest discover -s tests -v
venv/bin/python scripts/stress_test.py

# 显式真实取文及四路缓存并发
venv/bin/python scripts/stress_test.py --live --url http://127.0.0.1:8005 \
  --pdf-url https://aclanthology.org/2024.findings-acl.212.pdf --requests 4 --concurrency 4
```

CLI 从环境读取 SCHOLAR_ANALYSIS_ACCESS_TOKEN，不打印凭据。
真实分析必须另传 --analysis --allow-paid-analysis。
默认多个分析请求共享一个 ID 以验证不重复生成；明确需要独立生成时加
--distinct-analysis-ids。输出含语义检查、p50/p95、成功/退化/失败数量、
去重模型请求及费用上下界。验收事实见 [验证记录](docs/VALIDATION_20260912.md)。

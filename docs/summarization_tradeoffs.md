# Long Context Management Strategies Benchmark

## 1. Overview

This project evaluates three strategies for managing long-context information:

* Rolling Buffer
* Observation Masking
* Recursive Summarization

The evaluation is based on three metrics:

* Recall Accuracy
* Token Usage
* Execution Latency

---

## 2. Implemented Strategies

### Rolling Buffer

Maintains a fixed-size memory window containing recent observations.
Although efficient in memory usage, it may remove older important information.

### Observation Masking

Selectively retains relevant observations while reducing unnecessary context.
It aims to improve both accuracy and token efficiency.

### Recursive Summarization

Compresses previous observations into shorter summaries to handle large contexts.
However, summarization may introduce information loss.

---

## 3. Benchmark Results

| Strategy                | Accuracy | Tokens | Latency   |
| ----------------------- | -------- | ------ | --------- |
| Rolling Buffer          | 5/10     | 30     | 0.000063s |
| Observation Masking     | 10/10    | 13     | 0.000046s |
| Recursive Summarization | 8/10     | 34     | 0.000020s |

**Note:** Latency values were measured using a local Python implementation and represent function execution time only. They do not reflect the response latency of a real LLM API.

---

## 4. Results Analysis

Observation Masking achieved the highest recall accuracy (10/10) while using the smallest context size (13 tokens).

Recursive Summarization reduced context size effectively but caused some information loss, achieving 8/10 accuracy.

Rolling Buffer achieved the lowest accuracy (5/10) due to the removal of older observations from the memory window.

---

## 5. Final Recommendation

Based on the benchmark results, Observation Masking was selected as the preferred strategy for this workload because it provides the best balance between accuracy and context efficiency.

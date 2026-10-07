# GraphRAG Query Engine

- Source HTML: `raw/html/Microsoft - 2026 - GraphRAG Query Engine.html`
- Source SHA256: `7778e2af1f20a0115cc91ed60f00c3cd243b212b7f5e4ca53e929227cf7070b4`
- Source URL: https://microsoft.github.io/graphrag/query/overview/
- Generated from: `scripts/fetch_web_text.py`
- Extraction: `structured-html-v2` (headings, links, MathML/TeX and tables; figures require visual review)

## Extracted Text

<a id="source-section-0"></a>

<a id="source-section-1"></a>

# Query Engine 🔎

The Query Engine is the retrieval module of the GraphRAG library, and operates over completed [indexes](https://microsoft.github.io/graphrag/index/overview/).
It is responsible for the following tasks:

- [Local Search](https://microsoft.github.io/graphrag/query/overview/#local-search)

- [Global Search](https://microsoft.github.io/graphrag/query/overview/#global-search)

- [DRIFT Search](https://microsoft.github.io/graphrag/query/overview/#drift-search)

- Basic Search

- [Question Generation](https://microsoft.github.io/graphrag/query/overview/#question-generation)

<a id="source-section-2"></a>

## Local Search

Local search generates answers by combining relevant data from the AI-extracted knowledge-graph with text chunks of the raw documents. This method is suitable for questions that require an understanding of specific entities mentioned in the documents (e.g. What are the healing properties of chamomile?).

For more details about how Local Search works please refer to the [Local Search](https://microsoft.github.io/graphrag/query/local_search/) page.

<a id="source-section-3"></a>

## Global Search

Global search generates answers by searching over all AI-generated community reports in a map-reduce fashion. This is a resource-intensive method, but often gives good responses for questions that require an understanding of the dataset as a whole (e.g. What are the most significant values of the herbs mentioned in this notebook?).

More about this is provided on the [Global Search](https://microsoft.github.io/graphrag/query/global_search/) page.

<a id="source-section-4"></a>

## DRIFT Search

DRIFT Search introduces a new approach to local search queries by including community information in the search process. This greatly expands the breadth of the query’s starting point and leads to retrieval and usage of a far higher variety of facts in the final answer. This expands the GraphRAG query engine by providing a more comprehensive option for local search, which uses community insights to refine a query into detailed follow-up questions.

To learn more about DRIFT Search, please refer to the [DRIFT Search](https://microsoft.github.io/graphrag/query/drift_search/) page.

<a id="source-section-5"></a>

## Basic Search

GraphRAG includes a rudimentary implementation of basic vector RAG to make it easy to compare different search results based on the type of question you are asking. You can specify the top `k` text unit chunks to include in the summarization context.

<a id="source-section-6"></a>

## Question Generation

This functionality takes a list of user queries and generates the next candidate questions. This is useful for generating follow-up questions in a conversation or for generating a list of questions for the investigator to dive deeper into the dataset.

Information about how question generation works can be found at the [Question Generation](https://microsoft.github.io/graphrag/query/question_generation/) documentation page.

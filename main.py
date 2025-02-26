from llama_index.graph_stores.neo4j import Neo4jGraphStore
# from llama_index import ServiceContext, StorageContext
from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.query_engine import KnowledgeGraphQueryEngine
# from llama_index.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.schema import Document
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core import StorageContext, Settings

import time
time.sleep(18)
# 1. 连接 Neo4j
neo4j_graph_store = Neo4jGraphStore(
    username="neo4j",
    password="password",
    url="bolt://neo4j:7687",
)



# 2. 创建 StorageContext
storage_context = StorageContext.from_defaults(graph_store=neo4j_graph_store)

# 3. 配置 LLM（这里可以换成本地模型）
llm = OpenAI(model="gpt-4o-mini",api_key="xxx")  # 这里可以替换为本地模型

# service_context = ServiceContext.from_defaults(llm=llm)
Settings.llm = llm
Settings.chunk_size = 512

# 4. 创建知识图索引（GraphRAG）
documents = [
    Document(text="ChatGPT is an AI model developed by OpenAI.", id_="1"),
    Document(text="Neo4j is a graph database used for knowledge graphs.", id_="2"),
    Document(text="GraphRAG enhances retrieval using knowledge graphs.", id_="3"),
]
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    include_embeddings=True,
)

# 5. 构建查询引擎
query_engine = index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)

# 6. 查询 GraphRAG
query = "What is ChatGPT?"
response = query_engine.query(query)
print(response)
import os
import re
import json
import torch
import leidenalg
import igraph as ig
import pandas as pd
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import networkx as nx
from collections import defaultdict
import fitz  # PyMuPDF
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from networkx.readwrite import json_graph
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
# Load .env file
load_dotenv()

def load_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

EXTRACT_PROMPT = load_file("ENTITY_EXTRACTION_PROMPT.txt")
DESC_SUMMARIZE_PROMPT = load_file("ENTITY_DESCRIPTION_SUMMARIZATION_PROMPT.txt")
COMMUNITY_SUMMARIZATION_PROMPT = load_file("COMMUNITY_SUMMARIZATION_PROMPT.txt")
# Helper: simple token counter
def count_tokens(text: str) -> int:
    return len(text.split())
# 1. PDF Loading & Chunking
def load_and_split_pdf(PDF_PATH: str, chunk_size: int = 1000, chunk_overlap: int = 400):
    # ---------------------------
    # 1. Text Splitting
    # ---------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " "],
        is_separator_regex=False,
    )

    doc = fitz.open(PDF_PATH)
    chunks = []
    for pageno in range(doc.page_count):
        text = doc.load_page(pageno).get_text("text") or ""
        text = re.sub(r" +", " ", text)
        # normalize common ligatures/issues
        replacements = {"Ɵ": "ti", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi"}
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        docs = text_splitter.create_documents([text])
        for d in docs:
            chunks.append(d.page_content)
    return chunks


# 2. Dynamic Entity-Type Extraction
def extract_entity_types(client: Groq, initial_chunk: str, MODEL: str):
    # ---------------------------
    # 2. Dynamic Entity-Type Extraction via Llama
    # ---------------------------
    entity_type_prompt = (
        "Analyze the following document excerpt and list the main entity categories "
        "(e.g., Person, Organization, Date, Metric, Location, Financial Figure, etc.) "
        "that would be useful for knowledge graph extraction. Return as a JSON array only. "
        "Excerpt:\n" + initial_chunk
    )

    ent_resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": entity_type_prompt}],
    )

    raw_response = ent_resp.choices[0].message.content
    #print("[DEBUG] Entity-type response:\n", raw_response)

    # Extract JSON array from the response using regex
    try:
        match = re.search(r"\[.*?\]", raw_response, re.DOTALL)
        if match:
            entity_types = json.loads(match.group())
        else:
            raise ValueError("No JSON array found")
    except Exception as e:
        #print("[DEBUG] Failed to parse entity_types JSON, error:", e)
        entity_types = ["Person", "Organization", "Project", "Date", "Numeric Metric", "Location", "Algorithm"]

    return entity_types

# 3. Entity & Relationship Extraction
def extract_relations(client: Groq, chunks: list, entity_types: list, MODEL: str):
    records = []
    for idx, text in enumerate(tqdm(chunks, desc="Extract Entities")):
        tpl = EXTRACT_PROMPT.replace('{entity_types}', json.dumps(entity_types))
        prompt = tpl.replace('{input_text}', text)
        resp = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}])
        # DEBUG: raw model output per chunk
        #print(f"[DEBUG] Chunk {idx} raw output:\n", resp.choices[0].message.content)
        tuples = [t.strip() for t in resp.choices[0].message.content.split("##") if t.strip()]
        for tup in tuples:
            parts = tup.strip("()").split("<|>")
            if len(parts) >= 4:
                e1, e2, rel, score = parts[:4]
                print(f"[DEBUG] Parsed relationship: {e1} - {rel} -> {e2} (score={score})")
                records.append({
                    'chunk_id': f'chunk_{idx}',
                    'text': text,
                    'entity1': e1.strip().upper(),
                    'entity2': e2.strip().upper(),
                    'relationship': rel.strip(),
                    'score': int(score) if score.isdigit() else 1
                })
    return pd.DataFrame(records)

# 4. Initial Entity DECRIPTION GENERATION
def summarize_entities(client: Groq, df_rel: pd.DataFrame, MODEL: str):
    entity_contexts = defaultdict(set)
    for _, row in df_rel.iterrows():
        entity_contexts[row['entity1']].add(row['text'])
        entity_contexts[row['entity2']].add(row['text'])
    # (Optional) Convert to list later if needed
    entity_contexts = {k: list(v) for k, v in entity_contexts.items()}

    entity_desc = {}
    for entity, texts in tqdm(entity_contexts.items(), desc="Describing Entities"):
        sample = " ".join(texts[:2])  # sample first 2 chunks
        desc_prompt = (
            f"Provide a one-sentence description of '{entity}' based on the context: {sample}"
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": desc_prompt}],
        )
        entity_desc[entity] = resp.choices[0].message.content.strip()
    return entity_desc


# 5. Build Graph
def build_graph(df_rel: pd.DataFrame):
    # creating knowledge graph
    G = nx.Graph()

    # Add nodes with descriptions and associated text chunks
    for _, row in df_rel.iterrows():
        if G.has_node(row['entity1']):
            G.nodes[row['entity1']]['chunks'].add(row['text'])
            G.nodes[row['entity1']]['description'].add(row['description1'])
        else:
            G.add_node(row['entity1'], description={row['description1']}, chunks={row['text']})
        
        if G.has_node(row['entity2']):
            G.nodes[row['entity2']]['chunks'].add(row['text'])
            G.nodes[row['entity2']]['description'].add(row['description2'])
        else:
            G.add_node(row['entity2'], description={row['description2']}, chunks={row['text']})

    # Add edges with relationships and scores
    for _, row in df_rel.iterrows():
        if G.has_edge(row['entity1'], row['entity2']):
            existing_data = G[row['entity1']][row['entity2']]
            existing_data['relationship'] += f"\n{row['relationship']}"
            existing_data['score'] = max(existing_data['score'], row['score'])
        else:
            G.add_edge(row['entity1'], row['entity2'], relationship=row['relationship'], score=row['score'])

    # Convert the set of text chunks to a list
    for node in G.nodes:
        G.nodes[node]['chunks'] = list(G.nodes[node]['chunks'])

    # Concat descriptions
    for node in G.nodes:
        G.nodes[node]['description'] = list(G.nodes[node]['description'])
    return G

# 6. Merge Similar Nodes
def merge_similar_nodes(G: nx.Graph, threshold: float = 0.9):
    # ---------------------------
    # Node Embeddings & Clustering
    # ---------------------------
    # Compute embeddings for each node’s (name + description)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    labels = []
    texts = []
    for n, data in G.nodes(data=True):
        labels.append(n)
        texts.append(f"Name: {n}\nDescription:\n" + "\n".join(data['description']))

    embeds = embed_model.encode(texts, convert_to_numpy=True)
    cos = cosine_similarity(embeds)
    np.fill_diagonal(cos, 0)

    # BFS clustering on cos ≥ 0.9
    clusters = []
    visited = set()
    for i in range(len(labels)):
        if i in visited:
            continue
        queue = [i]
        comp = []
        while queue:
            j = queue.pop(0)
            if j in visited:
                continue
            visited.add(j)
            comp.append(j)
            queue += list(np.where(cos[j] >= threshold)[0])
        clusters.append(comp)
    # ---------------------------
    # 5. Build merged graph G2
    # ---------------------------
    G2 = nx.Graph()

    for comp in clusters:
        if len(comp) == 1:
            # Single-node cluster: copy it directly
            n_label = labels[comp[0]]
            G2.add_node(n_label, **G.nodes[n_label])
        else:
            # Multi-node cluster: merge into one node
            merged = "; ".join(labels[i] for i in comp)

            # Deduplicate and merge descriptions
            descs = [
                "\n".join(G.nodes[labels[i]]["description"])
                for i in comp
            ]
            G2.add_node(
                merged,
                description="\n".join(set(descs)),
                chunks=set(),
            )

            # Aggregate all text chunks
            for i in comp:
                G2.nodes[merged]["chunks"].update(
                    G.nodes[labels[i]]["chunks"]
                )

            # Precompute the set of original labels in this cluster
            cluster_labels = {labels[i] for i in comp}

            # Merge edges: reconnect external edges to the merged node
            for i in comp:
                src = labels[i]
                for nbr in G.neighbors(src):
                    data = G[src][nbr]

                    # Skip edges within the same cluster (no self-loops)
                    if nbr in cluster_labels:
                        continue

                    # If an edge already exists, append relationship & update score
                    if G2.has_edge(merged, nbr):
                        ex = G2[merged][nbr]
                        ex["relationship"] += f"\n{data['relationship']}"
                        ex["score"] = max(ex["score"], data["score"])
                    else:
                        # Otherwise, create a new edge
                        G2.add_edge(
                            merged,
                            nbr,
                            relationship=data["relationship"],
                            score=data["score"],
                        )
    # After the merge‑loop, re‑add edges for singleton nodes
    for u, v, data in G.edges(data=True):
        if u in G2.nodes and v in G2.nodes:
            # If this edge wasn't already added, bring it over
            if not G2.has_edge(u, v):
                G2.add_edge(u, v, **data)


    # Replace original graph with the merged version
    G = G2
    return G 

# 7. Long Description Summarization
def summarize_long_descriptions(client: Groq, G: nx.Graph, MODEL: str, token_threshold: int = 300):
    # Summarize long node descriptions
    for node, data in G.nodes(data=True):
        # Get description or default to empty
        raw_desc = data.get('description', "")
        # If a list, join into a string
        full = "\n".join(raw_desc) if isinstance(raw_desc, list) else raw_desc
        if count_tokens(full) > token_threshold:
            prompt = DESC_SUMMARIZE_PROMPT.replace('{entity_name}', node)
            prompt = prompt.replace('{description_list}', json.dumps([full]))
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}]
            )
            G.nodes[node]['description'] = resp.choices[0].message.content.strip()
    return G

# 8. Community Detection & Summarization
def detect_and_summarize_communities(client: Groq, G: nx.Graph, MODEL: str):
    # ---------------------------
    #  6. Community Detection & Summarization
    # ---------------------------


    # Convert NetworkX graph to iGraph
    nx_g = G.copy()
    g_igraph = ig.Graph(directed=False)

    node_mapping = {node: idx for idx, node in enumerate(nx_g.nodes())}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}

    edges = [(node_mapping[u], node_mapping[v]) for u, v in nx_g.edges()]

    g_igraph.add_vertices(len(node_mapping))
    g_igraph.add_edges(edges)

    # Use Leiden algorithm to detect communities
    partition = leidenalg.find_partition(g_igraph, leidenalg.ModularityVertexPartition)

    node_cluster_map = defaultdict(list)
    cluster_nodes = defaultdict(list)

    deg = nx.degree_centrality(G)

    for idx, comm in enumerate(partition):
        for node in comm:
            node_name = reverse_mapping[node]
            importance = deg[node_name]  # Use degree centrality
            node_cluster_map[node_name].append((idx, importance))
            cluster_nodes[idx].append(node_name)

    community_summary = {}
    for cid, nodes in cluster_nodes.items():
        info = {'communityId': cid, 'nodes': [], 'relationships': []}
        for n in nodes:
            # Get description or default to empty
            desc = G.nodes[n].get('description', "")
            desc = desc if isinstance(desc, str) else "\n".join(desc)
            info['nodes'].append({'id': n, 'description': desc})
        for u, v, data in G.edges(data=True):
            if u in nodes and v in nodes:
                info['relationships'].append({
                    'start': u,
                    'end': v,
                    'description': data['relationship']
                })
        COMMUNITY_SUMMARIZATION_PROMPT = """
        You are given a community of entities and their relationships in the form of a JSON. Summarize the key aspects of this community, focusing on the types of entities, their descriptions, and how they are related. Don't make it too large.
        
        Community Information:
        {community_info}
        
        Provide a concise summary that captures the main theme and structure of this community.
        """

        prompt = COMMUNITY_SUMMARIZATION_PROMPT.replace('{community_info}', json.dumps(info))
        if len(prompt.split()) > 10000:
            prompt = ' '.join(prompt.split()[:10000])
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        community_summary[cid] = resp.choices[0].message.content

    # Attach community summaries to nodes
    for node in G.nodes:
        if node in node_cluster_map:
            summaries = [community_summary[cid] for cid, _ in sorted(
                node_cluster_map[node], key=lambda x: x[1], reverse=True
            )]
            G.nodes[node]['community_summaries'] = summaries
        else:
            G.nodes[node]['community_summaries'] = []
    return G 
def convert_sets_to_lists(graph: nx.Graph) -> nx.Graph:
    for _, data in graph.nodes(data=True):
        for key in data:
            if isinstance(data[key], set):
                data[key] = list(data[key])
    for _, _, data in graph.edges(data=True):
        for key in data:
            if isinstance(data[key], set):
                data[key] = list(data[key])
    return graph

# 9. Persistence & Indexing
def persist_outputs(G: nx.Graph, df_rel: pd.DataFrame, work_dir: str):
    os.makedirs(work_dir, exist_ok=True)
    df_rel.to_csv(os.path.join(work_dir, 'chunk_rel_next.csv'), index=False)
    G = convert_sets_to_lists(G)

    with open(os.path.join(work_dir, 'graph.json'), 'w') as f:
        json.dump(json_graph.node_link_data(G,edges="edges"), f)
        #json.dump(json_graph.node_link_data(G,edges="links"), f)

def build_chroma_index(G: nx.Graph, WORK_DIR: str):
    # Build Chroma index
    emb_func = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    docs = [
        Document(
            page_content=f"Name: {n}\nDescription: {G.nodes[n].get('description', '')}",
            metadata={'source': n}
        )
        for n in G.nodes
    ]
    db = Chroma.from_documents(documents=docs, embedding=emb_func,persist_directory=WORK_DIR)
    return db
#print(EXTRACT_PROMPT)

# Master Pipeline

def build_kg():
    # ---------------------------
    # Configuration
    # ---------------------------
    # Set these environment variables or edit here:
    PDF_PATH = os.getenv("PDF_PATH", "./apple_disease.pdf")
    WORK_DIR = os.getenv("WORK_DIR", ".")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
    # Initialize Groq client
    client = Groq(api_key=GROQ_API_KEY)
    print(PDF_PATH)
    chunks = load_and_split_pdf(PDF_PATH)
    #print(chunks)
    #print(len(chunks))
    entity_types = extract_entity_types(client, chunks[0], MODEL)
    #print(entity_types)
    df_rel = extract_relations(client, chunks, entity_types, MODEL)
    entity_desc = summarize_entities(client, df_rel, MODEL)
    # Attach descriptions back onto the DataFrame
    df_rel['description1'] = df_rel['entity1'].map(entity_desc)
    df_rel['description2'] = df_rel['entity2'].map(entity_desc)
    # write CSV
    csv_path = os.path.join(WORK_DIR, "chunk_rel_with_des.csv")
    df_rel.to_csv(csv_path, index=False)
    print(f"[INFO] Saved CSV with {len(df_rel)} records to {csv_path}")

    
    G = build_graph(df_rel)
    print("length before merge")
    print(len(G))
    G = merge_similar_nodes(G)
    print("length after merge")
    print(len(G))
    ### to visualize the graph
    from pyvis.network import Network

    net = Network(notebook=False, height="750px", width="100%", bgcolor="#ffffff", font_color="black")

    for node in G.nodes(data=True):
        net.add_node(node[0], label=node[0], title=node[1].get('description', ''))

    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, title=d.get('relationship', ''), value=d.get('score', 1))

    net.save_graph("graph2.html")
    #######################
    G = summarize_long_descriptions(client, G, MODEL)

    G = detect_and_summarize_communities(client, G, MODEL)
    ###########SAVING G FOR FUTURE USE ##########
    
    # Save graph as GraphML (human-readable and supports node attributes)
    #graph_path = os.path.join(WORK_DIR, "kg.graphml")
    #nx.write_graphml(G, graph_path)
    #print(f"[✓] Graph saved to {graph_path}")
    # to LOAD USE
    #G = nx.read_graphml("kg.graphml")

    ##########################################
    persist_outputs(G, df_rel, WORK_DIR)
    db = build_chroma_index(G, WORK_DIR)
    print("final length")
    print(len(G))
    return G, db 
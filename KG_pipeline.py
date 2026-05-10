import os
import re
import json
import torch
import leidenalg
import igraph as ig
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

# Access the key
#api_key = os.getenv("GROQ_API_KEY")
# ---------------------------
# Configuration
# ---------------------------
# Set these environment variables or edit here:
PDF_PATH = os.getenv("PDF_PATH", "./apple1.pdf")
WORK_DIR = os.getenv("WORK_DIR", ".")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
print(PDF_PATH)
# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Load prompt templates
def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

EXTRACT_PROMPT = load_file("./ENTITY_EXTRACTION_PROMPT.txt")
DESC_SUMMARIZE_PROMPT = load_file("./ENTITY_DESCRIPTION_SUMMARIZATION_PROMPT.txt")
PREDICTION_PROMPT = load_file("./PREDICTION_PROMPT.txt")
#print(EXTRACT_PROMPT)
# Helper: simple token counter
def count_tokens(text: str) -> int:
    return len(text.split())

# ---------------------------
# 1. Text Splitting
# ---------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400,
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
# ---------------------------
# 2. Dynamic Entity-Type Extraction via Llama
# ---------------------------
entity_type_prompt = (
    "Analyze the following document excerpt and list the main entity categories "
    "(e.g., Person, Organization, Date, Metric, Location, Financial Figure, etc.) "
    "that would be useful for knowledge graph extraction. Return as a JSON array only. "
    "Excerpt:\n" + chunks[0][:1000]
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

#print("[DEBUG] Final entity types:\n", entity_types)
extract_tpl = (EXTRACT_PROMPT
    .replace("{tuple_delimiter}", "<|>")
    .replace("{completion_delimiter}", "<|COMPLETE|>")
    .replace("{entity_types}", json.dumps(entity_types))
    .replace("{record_delimiter}", "##")
)
# ---------------------------
# 4. Entity & Relationship Extraction via Llama
# ---------------------------
records = []
for idx, text in enumerate(tqdm(chunks, desc="Extract Entities")):
    pid = f"Chunk_{idx}"
    prompt = extract_tpl.replace("{input_text}", text)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    # DEBUG: raw model output per chunk
    #print(f"[DEBUG] Chunk {idx} raw output:\n", resp.choices[0].message.content)
    tuples = [r.strip() for r in resp.choices[0].message.content.split("##") if r.strip()]
    for tup in tuples:
        parts = tup.strip('()').split("<|>")
        if len(parts) >= 4:
            e1, e2, rel, score = parts[:4]
            #print(f"[DEBUG] Parsed relationship: {e1} - {rel} -> {e2} (score={score})")
            records.append({
                "chunk_id": pid,
                "text": text,
                "entity1": e1.upper(),
                "entity2": e2.upper(),
                "relationship": rel,
                "score": int(score) if score.isdigit() else 1
            })
# Save raw
import pandas as pd
df_rel = pd.DataFrame(records)
# DEBUG: DataFrame head
#print("[DEBUG] Records DataFrame head:\n", df_rel.head())
# write CSV
csv_path = os.path.join(WORK_DIR, "chunk_rel.csv")
df_rel.to_csv(csv_path, index=False)
print(f"[INFO] Saved CSV with {len(df_rel)} records to {csv_path}")
#print(len(df_rel))
# ---------------------------
# 6. Entity Description Extraction (Pass 2)
# ---------------------------

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

# Attach descriptions back onto the DataFrame
df_rel['description1'] = df_rel['entity1'].map(entity_desc)
df_rel['description2'] = df_rel['entity2'].map(entity_desc)

# write CSV
csv_path = os.path.join(WORK_DIR, "chunk_rel_with_des.csv")
df_rel.to_csv(csv_path, index=False)
print(f"[INFO] Saved CSV with {len(df_rel)} records to {csv_path}")



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

#################################################

# ---------------------------
#  5. Node Embeddings & Clustering
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
        queue += list(np.where(cos[j] >= 0.9)[0])
    clusters.append(comp)
################################################
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
############################################################

from pyvis.network import Network

net = Network(notebook=False, height="750px", width="100%", bgcolor="#ffffff", font_color="black")

for node in G.nodes(data=True):
    net.add_node(node[0], label=node[0], title=node[1].get('description', ''))

for u, v, d in G.edges(data=True):
    net.add_edge(u, v, title=d.get('relationship', ''), value=d.get('score', 1))

net.save_graph("graph2.html")
######################################
# Summarize long node descriptions
# Summarize long node descriptions
for node, data in G.nodes(data=True):
    # Get description or default to empty
    raw_desc = data.get('description', "")
    # If a list, join into a string
    full = "\n".join(raw_desc) if isinstance(raw_desc, list) else raw_desc
    if count_tokens(full) > 300:
        prompt = DESC_SUMMARIZE_PROMPT.replace('{entity_name}', node)
        prompt = prompt.replace('{description_list}', json.dumps([full]))
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        G.nodes[node]['description'] = resp.choices[0].message.content.strip()

'''for node, d in G.nodes(data=True):
    full = "\n".join(d['description']) if isinstance(d['description'], list) else d['description']
    if count_tokens(full) > 300:
        prompt = DESC_SUMMARIZE_PROMPT.replace('{entity_name}', node)
        prompt = prompt.replace('{description_list}', json.dumps([full]))
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        G.nodes[node]['description'] = resp.choices[0].message.content.strip()'''
######################################
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
        desc = G.nodes[n]['description']
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
########################################
# ---------------------------
#  7. Persist Graph & Vector Store
# ---------------------------
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

# Convert sets to lists or strings for JSON serialization
for n in G.nodes:
    if isinstance(G.nodes[n].get('chunks'), set):
        G.nodes[n]['chunks'] = list(G.nodes[n]['chunks'])
    if isinstance(G.nodes[n].get('description'), set):
        G.nodes[n]['description'] = "\n".join(G.nodes[n]['description'])

# Save KG JSON
with open(os.path.join(WORK_DIR, 'graph.json'), 'w') as f:
    json.dump(json_graph.node_link_data(G), f)

# Build Chroma index
emb_func = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
docs = [
    Document(
        page_content=f"Name: {n}\nDescription: {G.nodes[n]['description']}",
        metadata={'source': n}
    )
    for n in G.nodes
]
db = Chroma.from_documents(documents=docs, embedding=emb_func,persist_directory=WORK_DIR)
######################################
# Load prediction prompt from file
#with open("PREDICTION_PROMPT.txt") as pf:
    #PREDICTION_PROMPT = pf.read()

# ---------------------------
#  8. Querying & Response Generation Function
# ---------------------------
def generate_summary(
    db,
    G,
    client,
    MODEL: str,
    PREDICTION_PROMPT: str,
    query: str,
    topk_nodes: int = 2,
    topk_internal: int = 2,
    topk_external: int = 2,
    topk_chunks: int = 1
) -> str:
    """
    Retrieve relevant nodes and chunks from the KG, build context,
    call the LLM via client, and return the generated summary answer.
    """
    # Retrieve top-k relevant nodes
    docs = db.similarity_search(query, k=topk_nodes)
    node_ids = [d.metadata['source'] for d in docs]
    node_set = set(node_ids)

    # Keyword-based top-k chunk selection
    # Flatten all chunks for retrieved nodes
    all_chunks = [(nid, chunk) for nid in node_set for chunk in G.nodes[nid]['chunks']]
    # Prepare query terms
    q_terms = set(query.lower().split())
    # Score chunks by term overlap
    scored = sorted(
        (
            (len(q_terms & set(chunk.lower().split())), nid, chunk)
            for nid, chunk in all_chunks
        ),
        key=lambda x: x[0],
        reverse=True
    )[:topk_chunks]
    # Extract best chunks
    chunks = [c for _, _, c in scored]

    # Extract relationships
    internal_rels, external_rels = [], []
    for u, v, data in G.edges(data=True):
        rel_info = {"start": u, "end": v, "description": data['relationship'], "score": data['score']}
        if u in node_set and v in node_set:
            internal_rels.append(rel_info)
        elif u in node_set or v in node_set:
            external_rels.append(rel_info)

    # Limit top relationships
    internal_rels = sorted(internal_rels, key=lambda x: x['score'], reverse=True)[:topk_internal]
    external_rels = sorted(external_rels, key=lambda x: x['score'], reverse=True)[:topk_external]

    # Prepare node info
    nodes_set_for_info = set(r['start'] for r in internal_rels + external_rels) | set(r['end'] for r in internal_rels + external_rels)
    nodes_info = [
        {"id": nid, "description": G.nodes[nid]['description']}
        for nid in nodes_set_for_info
    ]

        # Community summaries (with node labels)
    comm_summaries = []
    for nid in node_set:
        cs = G.nodes[nid].get('community_summaries', [])
        if cs:
            # prefix each summary with its node identifier
            comm_summaries.append(f"• {nid}: {cs[0]}")

        # Build context string
    context_str = f"""
NODE DESCRIPTIONS:
{chr(10).join(f"• {n['id']}: {n['description']}" for n in nodes_info)}

CHUNKS:
{chr(10).join(chunks)}

INTERNAL RELATIONSHIPS:
{json.dumps(internal_rels, indent=2)}

EXTERNAL RELATIONSHIPS:
{json.dumps(external_rels, indent=2)}

COMMUNITY SUMMARIES:
{chr(10).join(comm_summaries)}
"""

    # Fill and send prediction prompt
    filled = (
        PREDICTION_PROMPT
        .replace("{context}", context_str)
        .replace("{question}", query)
    )
    #print(filled)
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[{"role": "user", "content": filled}]
    )
    return resp.choices[0].message.content


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # 1. Reconstruct MobileNetV3 architecture and load state_dict
    from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
    # Instantiate model with same base weights
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    # Reset classifier to match training setup (6 classes)
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    model.classifier[-2] = torch.nn.Dropout(p=0.3, inplace=True)
    num_classes = 6  # ensure matches training
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes, bias=True)
    model.classifier[-1].apply(init_weights)
    
    # Load the saved state_dict from Kaggle input
    state_dict_path = "./acc_model.pth"
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 2. Define preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 3. Classification helper
    class_names = ["complex", "healthy", "mildew", "rust", "scab", " Frogeye Leaf Spot"]
    def classify_leaf(image_path):
        img = Image.open(image_path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)  # move input to same device as model
        with torch.no_grad():
            logits = model(x)
        idx = logits.argmax(dim=-1).item()
        return class_names[idx]  # e.g. "Apple Scab", "Healthy"  # e.g. "Apple Scab", "Healthy"

    # 4. Simulate farmer upload and query input
    image_path = "./8d9d999d2ec52c29.jpg"  # replace with actual input path
    query_text = "how to prevent this disease?"
    
    # 5. Classify image
    disease_label = classify_leaf(image_path)
    print(f"Predicted disease: {disease_label}")

    # 6. Augment query with disease label and run RAG
    full_query = f"[Disease: {disease_label}] {query_text}"
    answer = generate_summary(
        db=db,
        G=G,
        client=client,
        MODEL=MODEL,
        PREDICTION_PROMPT=PREDICTION_PROMPT,
        query=full_query,
        topk_nodes=2,
        topk_internal=2,
        topk_external=2,
        topk_chunks=1
    )
    print("Answer:", answer)


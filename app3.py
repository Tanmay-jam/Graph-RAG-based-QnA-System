# app.py
# Streamlit app: multi-turn Q&A, disease classification & comparisons

import os
import streamlit as st
import torch
from PIL import Image
from groq import Groq
from KG_pipeline1 import build_kg
from torchvision import transforms
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
import json
# Load existing graph from JSON
import networkx as nx
from networkx.readwrite import json_graph
from langchain_community.vectorstores import Chroma as _Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
def load_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

PREDICTION_PROMPT = load_file("PREDICTION_PROMPT.txt")

# --- RAG query function ---
def generate_summary(db, G, client, model_name, prompt_text,
                     topk_nodes=2, topk_internal=2, topk_external=2, topk_chunks=2):
    docs = db.similarity_search(prompt_text, k=topk_nodes)
    node_ids = [d.metadata['source'] for d in docs]
    node_set = set(node_ids)
    all_chunks = [
    (n, c)
    for n in node_set
    for c in G.nodes[n].get('chunks', [])]

    terms = set(prompt_text.lower().split())
    scored = sorted(
        ((len(terms & set(c.lower().split())), n, c) for n, c in all_chunks),
        key=lambda x: x[0], reverse=True
    )[:topk_chunks]
    chunks = [c for _,_,c in scored]
    internal, external = [], []
    for u,v,d in G.edges(data=True):
        info = {'start':u,'end':v,'description':d['relationship'],'score':d['score']}
        if u in node_set and v in node_set:
            internal.append(info)
        elif u in node_set or v in node_set:
            external.append(info)
    internal = sorted(internal, key=lambda x: x['score'], reverse=True)[:topk_internal]
    external = sorted(external, key=lambda x: x['score'], reverse=True)[:topk_external]
    # Prepare node info
    nodes_set_for_info = set(r['start'] for r in internal + external) | set(r['end'] for r in internal + external)
    descs = [f"• {n}: {G.nodes[n].get('description', '')}" for n in nodes_set_for_info]
    comm = [f"• {n}: {G.nodes[n].get('community_summary','')}" for n in node_set]
    context = (
        "\n\nCHUNKS:\n" + "\n".join(chunks) +
        "NODE DESCRIPTIONS:\n" + "\n".join(descs) +
        "\n\nINTERNAL RELS:\n" + json.dumps(internal, indent=2) +
        "\n\nEXTERNAL RELS:\n" + json.dumps(external, indent=2) +
        "\n\nCOMMUNITY:\n" + "\n".join(comm)
    )
    filled = PREDICTION_PROMPT.replace('{context}', context).replace('{question}', prompt_text)
    print("#############################")
    #print(filled)
    resp = client.chat.completions.create(
        model=model_name, temperature=0.1,
        messages=[{'role':'user','content':filled}]
    )
    return resp.choices[0].message.content

# --- Initialize resources ---
@st.cache_resource
def init_resources():
    pdf_path   = os.getenv('PDF_PATH','./apple1.pdf')
    work_dir   = os.getenv('WORK_DIR','.')
    model_name = os.getenv('LLM_MODEL','llama-3.1-8b-instant')

    # Check for existing persisted KG in JSON and Chroma DB
    json_path = os.path.join(work_dir, 'graph.json')
    # Chroma uses sqlite file named 'chroma.sqlite3' by default
    db_file   = os.path.join(work_dir, 'chroma.sqlite3')  # default Chroma sqlite file
    if os.path.exists(json_path) and os.path.exists(db_file):
        # Load existing graph and vector store
        import networkx as nx
        from networkx.readwrite import json_graph
        from langchain_community.vectorstores import Chroma as _Chroma
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Rename "edges" to "links" for NetworkX compatibility
        if "edges" in data and "links" not in data:
            data["links"] = data.pop("edges")
        G = json_graph.node_link_graph(data, directed=False)
        # ✅ Provide embedding function during loading
        emb_func = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        db = _Chroma(persist_directory=work_dir, embedding_function=emb_func)
    else:
        # Build KG and vector store if not persisted
        G, db = build_kg()

    # Initialize Groq client
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    # Leaf classifier loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf    = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    clf.classifier[-2] = torch.nn.Dropout(p=0.3)
    in_f    = clf.classifier[-1].in_features
    clf.classifier[-1] = torch.nn.Linear(in_f,6)
    clf.load_state_dict(
        torch.load(os.path.join(work_dir,'mobilenetv3_leaf_model.pth'), map_location=device)
    )
    clf.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    return G, db, client, clf, preprocess, device, model_name

# --- App state ---
if 'history' not in st.session_state:
    st.session_state.history = []  # list of (speaker,message)

st.title('🍎 Leaf Disease Q&A')
G, db, client, clf, preprocess, device, model_name = init_resources()

# Sidebar
st.sidebar.header('Your Input')
use_img    = st.sidebar.checkbox('Include leaf image?')
upfile     = st.sidebar.file_uploader('Upload leaf image', type=['jpg','png']) if use_img else None
question   = st.sidebar.text_input('Your question')
ask_button = st.sidebar.button('Ask')

if ask_button and question:
    speaker = 'Farmer'
    if use_img and upfile:
        img = Image.open(upfile).convert('RGB')
        t   = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            idx = clf(t).argmax().item()
        labels = ['complex','healthy','Powdery Mildew','Cedar-Apple Rust ','Apple Scab','Frogeye Leaf Spot ']
        query = f"[Disease: {labels[idx]}] {question}"
    else:
        query = question
    st.session_state.history.append((speaker, query))
    ans = generate_summary(db, G, client, model_name, query)
    st.session_state.history.append(('Machine', ans))

# Render chat
for i,(sp,m) in enumerate(st.session_state.history, start=1):
    st.markdown(f"**Turn {i} | {sp}:** {m}")
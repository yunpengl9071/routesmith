"""
Run RouteLLM with cached smaller dataset.
"""
import os
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENROUTER_API_KEY', '')
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

import json
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import expit
from datasets import load_from_disk

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

STRONG_MODEL = "openai/gpt-4o-mini"
WEAK_MODEL = "meta-llama/llama-3.1-8b-instruct"

# Load cached training data
print("Loading cached training data...")
train_ds = load_from_disk("/home/yliulupo/projects/routesmith/report/routellm_train_5k")
print(f"Loaded {len(train_ds)} training examples")

# Get embeddings (these should be pre-cached or use cached embeddings)
print("Loading pre-computed embeddings...")
try:
    from datasets import load_dataset
    emb_ds = load_dataset("routellm/arena_battles_embeddings", split="train[:5000]")
    train_embeddings = np.array(emb_ds.to_dict()["embeddings"])
    print(f"Loaded {len(train_embeddings)} embeddings")
except Exception as e:
    print(f"Could not load embeddings: {e}")
    print("Computing embeddings on the fly...")
    # Would need to compute here but that's slow
    exit(1)

# Get preferences
train_preferences = []
for i in range(len(train_ds)):
    row = train_ds[i]
    # preference = 1 if strong model (gpt-4) won, 0 otherwise
    if row['winner_model_a'] == 1:
        train_preferences.append(1)
    elif row['winner_model_b'] == 1:
        train_preferences.append(0)
    else:
        train_preferences.append(0.5)  # tie

print(f"Training data: {sum(1 for p in train_preferences if p==1)} strong wins, {sum(1 for p in train_preferences if p==0)} weak wins")

def bradley_terry_score(query_embedding, train_embeddings, train_preferences, gamma=10.0):
    similarities = cosine_similarity([query_embedding], train_embeddings)[0]
    weights = gamma ** similarities
    
    strong_votes = np.sum(weights * np.array(train_preferences))
    weak_votes = np.sum(weights * (1 - np.array(train_preferences)))
    
    score = strong_votes - weak_votes
    total_weight = np.sum(weights)
    if total_weight > 0:
        score = score / total_weight
    
    return expit(score)

def get_embedding(text):
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

# Load test queries
with open("/home/yliulupo/projects/routesmith/report/test_queries_50.json") as f:
    test_queries = json.load(f)

print(f"\nRunning benchmark on {len(test_queries)} queries...")

results = []
strong_count = 0
weak_count = 0
total_cost = 0

strong_cost_per_1k = 0.15
weak_cost_per_1k = 0.0002

for i, item in enumerate(test_queries):
    q = item["query"]
    print(f"Query {i+1}/{len(test_queries)}: {q[:40]}...")
    
    # Get query embedding
    query_emb = get_embedding(q)
    
    # Get win probability
    win_prob = bradley_terry_score(query_emb, train_embeddings, train_preferences)
    print(f"  Win prob: {win_prob:.3f}")
    
    if win_prob > 0.5:
        route = "strong"
        strong_count += 1
        total_cost += strong_cost_per_1k
    else:
        route = "weak"
        weak_count += 1
        total_cost += weak_cost_per_1k
    
    results.append({
        "query": q,
        "routed_to": route,
        "win_probability": win_prob
    })
    print(f"  -> Routed to: {route}")

# Summary
print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Total: {len(test_queries)}, Strong: {strong_count} ({100*strong_count/len(test_queries):.1f}%), Weak: {weak_count}")
print(f"Cost (all strong): ${strong_cost_per_1k * len(test_queries):.4f}")
print(f"Cost (routed): ${total_cost:.4f}")
print(f"Cost savings: {100*(1 - total_cost/(strong_cost_per_1k * len(test_queries))):.1f}%")

# Save
with open("/home/yliulupo/projects/routesmith/report/routellm_real_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to routellm_real_results.json")

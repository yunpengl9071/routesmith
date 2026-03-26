"""
Similarity-Weighted Ranking Router (RouteLLM-style)
Implementation based on the RouteLLM paper: https://arxiv.org/abs/2406.18665

This implements the SW Ranking approach:
1. Embed queries using text-embedding-3-small
2. Compute similarity to training queries
3. Use Bradley-Terry model to predict win probability

IMPORTANT CAVEAT: This uses SYNTHETIC training data (20 examples), not the real 
RouteLLM model which was trained on 80K+ Chatbot Arena samples. Results are not 
directly comparable to the published RouteLLM numbers.
"""
import os
import json
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import expit  # sigmoid function

# Configure OpenAI client for OpenRouter
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Model configuration
STRONG_MODEL = "openai/gpt-4o-mini"  # Strong model (GPT-4 level)
WEAK_MODEL = "meta-llama/llama-3.1-8b-instruct"  # Weak model (cheaper)

def get_embedding(text, client=client):
    """Get embedding for text using OpenRouter's embedding endpoint."""
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def bradley_terry_score(query_embedding, train_embeddings, train_preferences, xi_s=1.0, xi_w=1.0, gamma=10.0):
    """
    Compute Bradley-Terry score for strong model winning.
    Following RouteLLM paper: P(strong) = sigmoid(xi_s * weighted_strong_votes - xi_w * weighted_weak_votes)
    
    The intuition: if query is more similar to queries where strong model won, route to strong.
    
    Args:
        query_embedding: embedding of the input query
        train_embeddings: list of embeddings for training queries
        train_preferences: list of preference labels (1 = strong wins, 0 = weak wins)
        xi_s, xi_w: Bradley-Terry coefficients
        gamma: similarity weight exponent (default 10 from paper)
    
    Returns:
        Probability of strong model winning
    """
    if len(train_embeddings) == 0:
        return 0.5
    
    train_embeddings = np.array(train_embeddings)
    similarities = cosine_similarity([query_embedding], train_embeddings)[0]
    
    # Weight = gamma^(similarity) - from RouteLLM paper
    weights = gamma ** similarities
    
    # Compute weighted votes for each model
    # Strong model gets positive votes weighted by similarity to queries it won
    strong_votes = np.sum(weights * np.array(train_preferences))
    # Weak model gets positive votes weighted by similarity to queries it won
    weak_votes = np.sum(weights * (1 - np.array(train_preferences)))
    
    # Score: positive if more similar to strong-winning queries
    score = xi_s * strong_votes - xi_w * weak_votes
    
    # Normalize by total weight to keep in reasonable range
    total_weight = np.sum(weights)
    if total_weight > 0:
        score = score / total_weight
    
    return expit(score)

def load_training_data():
    """
    Load synthetic training data based on Chatbot Arena format.
    Since we can't load RouteLLM's data directly, we create a representative dataset.
    """
    # Balanced training data representing different query types
    training_queries = [
        # Simple factual - weak model sufficient (preference = 0)
        {"query": "What is the capital of France?", "preference": 0},
        {"query": "Who was the first president of the US?", "preference": 0},
        {"query": "What year did WW2 end?", "preference": 0},
        {"query": "What is 2+2?", "preference": 0},
        {"query": "What is H2O?", "preference": 0},
        {"query": "What planet is closest to the sun?", "preference": 0},
        {"query": "Write hello world in Python", "preference": 0},
        {"query": "What is 15 * 23?", "preference": 0},
        {"query": "List the colors of the rainbow", "preference": 0},
        {"query": "What is the largest ocean?", "preference": 0},
        
        # Complex tasks - strong model better (preference = 1)
        {"query": "Write a Python function to implement quicksort", "preference": 1},
        {"query": "Debug this JavaScript code", "preference": 1},
        {"query": "Create a REST API with authentication", "preference": 1},
        {"query": "Solve this differential equation", "preference": 1},
        {"query": "Prove this theorem", "preference": 1},
        {"query": "Explain quantum entanglement in detail", "preference": 1},
        {"query": "Write a complex regex pattern", "preference": 1},
        {"query": "Implement a neural network from scratch", "preference": 1},
        {"query": "Explain the theory of relativity", "preference": 1},
        {"query": "Write a distributed system in Go", "preference": 1},
    ]
    return training_queries

def load_training_embeddings(training_data, client):
    """Pre-compute and cache training embeddings."""
    print(f"  Embedding {len(training_data)} training queries (caching)...")
    train_embeddings = []
    for d in training_data:
        emb = get_embedding(d["query"], client)
        train_embeddings.append(emb)
    return np.array(train_embeddings)

def route_query(query, training_data, train_embeddings, threshold=0.5, client=client):
    """
    Decide whether to route to strong or weak model.
    
    Args:
        query: User query
        training_data: List of dicts with 'query' and 'preference' keys
        train_embeddings: Pre-computed embeddings for training data
        threshold: Cost threshold (if win_prob > threshold, use strong model)
    
    Returns:
        'strong' or 'weak'
    """
    # Get query embedding
    query_emb = get_embedding(query, client)
    
    train_prefs = [d["preference"] for d in training_data]
    
    # Compute win probability using cached embeddings
    win_prob = bradley_terry_score(query_emb, train_embeddings, train_prefs)
    
    print(f"  Win probability: {win_prob:.3f}, threshold: {threshold}")
    
    if win_prob > threshold:
        return "strong", win_prob
    else:
        return "weak", win_prob

def call_model(query, model_name, client=client):
    """Call the LLM with the query."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

def run_benchmark(queries, threshold=0.5):
    """Run the routing benchmark."""
    print(f"Loading training data...")
    training_data = load_training_data()
    
    # Pre-compute training embeddings ONCE
    print(f"Pre-computing training embeddings...")
    train_embeddings = load_training_embeddings(training_data, client)
    
    results = []
    strong_count = 0
    weak_count = 0
    total_cost = 0
    
    # Estimate costs (per 1M tokens, approximate)
    strong_cost_per_1k = 0.15  # gpt-4o-mini
    weak_cost_per_1k = 0.0002  # llama 3.1 8b
    
    print(f"\nRunning benchmark on {len(queries)} queries...")
    print(f"Threshold: {threshold}")
    print(f"Strong model: {STRONG_MODEL}")
    print(f"Weak model: {WEAK_MODEL}")
    print()
    
    for i, item in enumerate(queries):
        q = item["query"] if isinstance(item, dict) else item
        print(f"Query {i+1}/{len(queries)}: {q[:50]}...")
        
        # Route the query (using cached embeddings)
        route, win_prob = route_query(q, training_data, train_embeddings, threshold)
        
        if route == "strong":
            strong_count += 1
            total_cost += strong_cost_per_1k
            model_used = STRONG_MODEL
        else:
            weak_count += 1
            total_cost += weak_cost_per_1k
            model_used = WEAK_MODEL
        
        results.append({
            "query": q,
            "routed_to": route,
            "win_probability": win_prob,
            "model_used": model_used
        })
        print(f"  -> Routed to: {route}")
        print()
    
    # Summary
    print("=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total queries: {len(queries)}")
    print(f"Routed to strong: {strong_count} ({100*strong_count/len(queries):.1f}%)")
    print(f"Routed to weak: {weak_count} ({100*weak_count/len(queries):.1f}%)")
    print(f"Estimated cost (strong only): ${strong_cost_per_1k * len(queries):.4f}")
    print(f"Estimated cost (with routing): ${total_cost:.4f}")
    print(f"Cost savings: {100*(1 - total_cost/(strong_cost_per_1k * len(queries))):.1f}%")
    
    return results

if __name__ == "__main__":
    # Load benchmark queries
    benchmark_file = "/home/yliulupo/projects/routesmith/report/test_queries_50.json"
    try:
        with open(benchmark_file) as f:
            queries = json.load(f)
        print(f"Loaded {len(queries)} benchmark queries")
    except:
        queries = [
            {"query": "What is the capital of France?", "category": "factual"},
            {"query": "Write a Python function to reverse a string", "category": "coding"},
            {"query": "Explain quantum entanglement in simple terms", "category": "science"},
            {"query": "Calculate 123 * 456", "category": "math"},
            {"query": "Debug this code: for i in range(10): print(i", "category": "coding"},
        ]
        print(f"Using default {len(queries)} queries")
    
    results = run_benchmark(queries, threshold=0.5)
    
    # Save results
    output_file = "/home/yliulupo/projects/routesmith/report/routellm_sw_ranking_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

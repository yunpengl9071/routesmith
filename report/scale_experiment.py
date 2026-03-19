#!/usr/bin/env python3
"""
Scale-Up RouteSmith Experiment
60+ queries, 5 models, LLM-judge validation, cost tracking
Max budget: $30
"""

import os
import json
import time
import asyncio
import aiohttp
import random
from datetime import datetime
from collections import defaultdict

API_KEY = os.environ.get("OPENROUTER_API_KEY", "REDACTED_OPENROUTER_KEY_2")

# Model configurations
MODELS = {
    "nemotron": {
        "id": "nvidia/nemotron-3-nano-30b-a3b:free",
        "tier": "Free",
        "name": "Nemotron 3 Nano"
    },
    "phi4": {
        "id": "microsoft/phi-4",
        "tier": "Budget",
        "name": "Microsoft Phi-4"
    },
    "gpt4o_mini": {
        "id": "openai/gpt-4o-mini",
        "tier": "Standard",
        "name": "GPT-4o Mini"
    },
    "deepseek": {
        "id": "deepseek/deepseek-v3.2",
        "tier": "Standard",
        "name": "DeepSeek V3.2"
    },
    "gpt4o": {
        "id": "openai/gpt-4o",
        "tier": "Premium",
        "name": "GPT-4o"
    }
}

# Realistic customer support queries (60+)
QUERIES = [
    # Account & Billing (15)
    ("How do I reset my password?", "simple"),
    ("What is your refund policy?", "simple"),
    ("How do I cancel my subscription?", "simple"),
    ("Can I change my billing address?", "simple"),
    ("Why was I charged twice?", "moderate"),
    ("How do I update my credit card?", "simple"),
    ("What's my current plan level?", "simple"),
    ("How do I download my invoice?", "simple"),
    ("Can I get a receipt for my purchase?", "simple"),
    ("How do I upgrade to premium?", "simple"),
    ("What payment methods do you accept?", "simple"),
    ("How do I request a refund?", "moderate"),
    ("My payment failed, what should I do?", "moderate"),
    ("Can I switch from monthly to annual billing?", "simple"),
    ("Where can I see my billing history?", "simple"),
    
    # Technical Support (20)
    ("The app keeps crashing on startup", "moderate"),
    ("I can't log into my account", "moderate"),
    ("How do I enable two-factor authentication?", "simple"),
    ("My API requests are returning 500 errors", "complex"),
    ("How do I set up OAuth refresh token rotation?", "complex"),
    ("Webhook signatures don't match", "complex"),
    ("How do I implement cursor-based pagination?", "complex"),
    ("My async jobs are timing out", "complex"),
    ("CORS preflight requests failing", "complex"),
    ("Database connection pool exhausted", "complex"),
    ("How do I integrate your API with Node.js?", "moderate"),
    ("Your SDK doesn't work with Python 3.12", "moderate"),
    ("Memory leak in production server", "complex"),
    ("SSL certificate error on my domain", "complex"),
    ("Rate limiting too aggressive", "moderate"),
    ("How do I increase my API rate limit?", "moderate"),
    ("WebSocket connection drops after 5 minutes", "complex"),
    ("Missing fields in API response", "moderate"),
    ("How do I authenticate with API key vs OAuth?", "moderate"),
    ("File upload failing above 10MB", "moderate"),
    
    # Product & Features (15)
    ("What features are included in the free plan?", "simple"),
    ("Does your product support dark mode?", "simple"),
    ("How do I export my data to CSV?", "simple"),
    ("Can I create custom dashboards?", "moderate"),
    ("What's the difference between Basic and Pro?", "simple"),
    ("Do you offer team collaboration features?", "simple"),
    ("How do I invite team members?", "simple"),
    ("Can I set up role-based permissions?", "moderate"),
    ("Does the mobile app have all features?", "simple"),
    ("How do I set up webhooks for notifications?", "moderate"),
    ("Can I integrate with Slack?", "simple"),
    ("Do you have an API for bulk operations?", "moderate"),
    ("How do I create automated workflows?", "complex"),
    ("What's your uptime guarantee?", "simple"),
    ("Do you offer white-label options?", "complex"),
    
    # Troubleshooting (15)
    ("Login page shows blank screen", "moderate"),
    ("Reports not loading correctly", "moderate"),
    ("Search results are inaccurate", "moderate"),
    ("My notifications aren't working", "moderate"),
    ("Data not syncing between devices", "moderate"),
    ("Charts display incorrectly on mobile", "moderate"),
    ("Can't upload images to profile", "moderate"),
    ("Email notifications delayed", "moderate"),
    ("Export file is corrupted", "moderate"),
    ("Filter options not working", "simple"),
    ("Page loads very slowly", "moderate"),
    ("Buttons not clickable on Safari", "moderate"),
    ("Drop-down menus not opening", "simple"),
    ("Calendar shows wrong timezone", "moderate"),
    ("PDF generation fails with special characters", "complex"),
]

# Sample 20% for LLM judge
JUDGE_SAMPLE_SIZE = 12  # 20% of 60

# Cost tracking
total_cost = 0
cost_by_model = defaultdict(float)

async def call_model(session, model_key, query, retries=2):
    """Call OpenRouter API"""
    global total_cost
    
    model_info = MODELS[model_key]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; RouteSmith/1.0)",
    }
    payload = {
        "model": model_info["id"],
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 250,
    }
    
    for attempt in range(retries):
        start = time.time()
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                latency = time.time() - start
                data = await resp.json()
                
                if resp.status == 200 and "choices" in data and len(data["choices"]) > 0:
                    response = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    cost = usage.get("cost", 0)
                    
                    total_cost += cost
                    cost_by_model[model_key] += cost
                    
                    return {
                        "success": True,
                        "response": response,
                        "latency": round(latency, 2),
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "cost": cost,
                    }
                elif resp.status == 429:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return {"success": False, "error": data.get("error", {}), "status": resp.status}
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(1)
                continue
            return {"success": False, "error": str(e)}
    
    return {"success": False, "error": "Max retries exceeded"}

async def llm_judge(session, query, responses):
    """Use Claude Haiku to judge response quality"""
    judge_prompt = f"""You are an expert evaluator judging customer support responses.
Rate each response on a scale of 1-10 based on:
- Relevance: Does it directly answer the user's question?
- Completeness: Does it provide enough detail?
- Clarity: Is it easy to understand?

Query: {query}

Responses to evaluate:"""
    
    for model_key, resp in responses.items():
        judge_prompt += f"\n\n{model_key.upper()} ({MODELS[model_key]['name']}):\n{resp.get('response', 'NO RESPONSE')[:500]}"
    
    judge_prompt += "\n\nOutput your scores as JSON with format: {\"model_key\": score} (1-10 integer scores only)"
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "anthropic/claude-3-haiku:free",
        "messages": [{"role": "user", "content": judge_prompt}],
        "max_tokens": 300,
    }
    
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            data = await resp.json()
            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
                # Parse JSON from response
                import re
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    scores = json.loads(json_match.group())
                    return {k: int(v) for k, v in scores.items()}
    except Exception as e:
        print(f"Judge error: {e}")
    
    return None

async def run_experiment():
    """Run the full scale experiment"""
    results = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "num_queries": len(QUERIES),
            "models": list(MODELS.keys()),
        },
        "responses": [],
        "judge_results": [],
        "costs": {},
        "summary": {}
    }
    
    # Select queries for LLM judge (20%)
    judge_indices = set(random.sample(range(len(QUERIES)), JUDGE_SAMPLE_SIZE))
    
    async with aiohttp.ClientSession() as session:
        for i, (query, complexity) in enumerate(QUERIES):
            print(f"\n[{i+1}/{len(QUERIES)}] {query[:50]}...")
            
            query_results = {
                "query": query,
                "complexity": complexity,
                "model_responses": {}
            }
            
            # Get responses from all models
            for model_key in MODELS:
                print(f"  → {model_key}", end=" ", flush=True)
                resp = await call_model(session, model_key, query)
                
                if resp.get("success"):
                    print(f"✓ ${resp['cost']:.4f}")
                else:
                    print(f"✗")
                
                query_results["model_responses"][model_key] = resp
            
            results["responses"].append(query_results)
            
            # LLM judge for sampled queries
            if i in judge_indices:
                print(f"  ⚖️  Running LLM judge...")
                judge_scores = await llm_judge(session, query, query_results["model_responses"])
                if judge_scores:
                    query_results["judge_scores"] = judge_scores
                    results["judge_results"].append({
                        "query": query,
                        "scores": judge_scores
                    })
                    print(f"  ✓ Judge complete: {judge_scores}")
            
            # Progress report every 15 queries
            if (i + 1) % 15 == 0:
                print(f"\n{'='*50}")
                print(f"Progress: {i+1}/{len(QUERIES)} queries")
                print(f"Current spend: ${total_cost:.4f}")
                print(f"{'='*50}")
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.3)
    
    # Calculate summary statistics
    print("\n" + "="*60)
    print("CALCULATING SUMMARY STATISTICS...")
    
    for model_key, model_info in MODELS.items():
        scores = []
        latencies = []
        costs = []
        
        for qr in results["responses"]:
            resp = qr["model_responses"].get(model_key, {})
            if resp.get("success"):
                # Heuristic quality score
                response_text = resp.get("response", "").lower()
                query_text = qr["query"].lower()
                
                # Simple heuristic
                quality = 5  # base
                if any(q in response_text for q in query_text.split()[:3] if len(q) > 3):
                    quality += 2
                if len(resp.get("response", "")) > 80:
                    quality += 1
                if len(resp.get("response", "")) > 150:
                    quality += 1
                if "error" not in response_text:
                    quality += 1
                quality = min(10, max(1, quality))
                
                scores.append(quality)
                latencies.append(resp.get("latency", 0))
                costs.append(resp.get("cost", 0))
        
        results["summary"][model_key] = {
            "name": model_info["name"],
            "tier": model_info["tier"],
            "avg_heuristic_quality": round(sum(scores)/len(scores), 2) if scores else 0,
            "avg_latency": round(sum(latencies)/len(latencies), 2) if latencies else 0,
            "total_cost": round(sum(costs), 4),
            "successful_queries": len(scores),
        }
    
    results["costs"]["total"] = round(total_cost, 4)
    results["costs"]["by_model"] = dict(cost_by_model)
    
    # Process LLM judge results
    if results["judge_results"]:
        judge_summary = defaultdict(list)
        for jr in results["judge_results"]:
            for model_key, score in jr["scores"].items():
                judge_summary[model_key].append(score)
        
        results["summary"]["llm_judge_avg"] = {
            model: round(sum(scores)/len(scores), 2) if scores else 0
            for model, scores in judge_summary.items()
        }
    
    return results

def generate_report(results):
    """Generate markdown report"""
    report = []
    report.append("# RouteSmith Scale-Up Experiment Results")
    report.append(f"\n**Date:** {results['metadata']['date']}")
    report.append(f"**Queries Tested:** {results['metadata']['num_queries']}")
    report.append(f"**Models Tested:** {len(MODELS)}")
    report.append(f"**LLM-Judged Samples:** {len(results['judge_results'])} ({JUDGE_SAMPLE_SIZE} queries)")
    report.append(f"**Total API Cost:** ${results['costs']['total']:.4f}")
    report.append("")
    
    # Model comparison table (heuristic)
    report.append("## Model Comparison (Heuristic Quality Scores)")
    report.append("")
    report.append("| Model | Tier | Avg Quality | Avg Latency | Cost |")
    report.append("|-------|------|-------------|-------------|------|")
    
    for model_key, data in results["summary"].items():
        if model_key in MODELS:
            report.append(f"| {data['name']} | {data['tier']} | {data['avg_heuristic_quality']}/10 | {data['avg_latency']}s | ${data['total_cost']:.4f} |")
    
    report.append("")
    
    # LLM Judge results
    if "llm_judge_avg" in results["summary"]:
        report.append("## LLM-as-Judge Validation (Claude Haiku)")
        report.append("")
        report.append("| Model | LLM-Judge Score | Heuristic Score | Delta |")
        report.append("|-------|-----------------|-----------------|-------|")
        
        for model_key in MODELS:
            llm_score = results["summary"]["llm_judge_avg"].get(model_key, 0)
            heuristic = results["summary"][model_key]["avg_heuristic_quality"]
            delta = llm_score - heuristic
            sign = "+" if delta > 0 else ""
            report.append(f"| {MODELS[model_key]['name']} | {llm_score}/10 | {heuristic}/10 | {sign}{delta:.1f} |")
        
        report.append("")
    
    # Cost projections
    report.append("## Cost Savings Projections")
    report.append("")
    
    baseline = results["summary"]["gpt4o"]["total_cost"] / results["metadata"]["num_queries"]
    
    report.append("### Per-Query Costs")
    report.append("| Model | Cost/Query | vs Premium |")
    report.append("|-------|------------|------------|")
    
    for model_key in ["nemotron", "phi4", "gpt4o_mini", "deepseek", "gpt4o"]:
        data = results["summary"][model_key]
        per_query = data["total_cost"] / data["successful_queries"] if data["successful_queries"] else 0
        savings = (1 - per_query/baseline) * 100 if baseline > 0 else 0
        report.append(f"| {data['name']} | ${per_query:.4f} | {savings:.0f}% |")
    
    report.append("")
    report.append("### Projected Daily Costs (1K, 10K, 100K queries)")
    report.append("| Model | 1K/day | 10K/day | 100K/day |")
    report.append("|-------|--------|---------|----------|")
    
    for model_key in ["nemotron", "phi4", "gpt4o_mini", "deepseek", "gpt4o"]:
        data = results["summary"][model_key]
        per_query = data["total_cost"] / data["successful_queries"] if data["successful_queries"] else 0
        report.append(f"| {data['name']} | ${per_query*1000:.2f} | ${per_query*10000:.2f} | ${per_query*100000:.2f} |")
    
    report.append("")
    
    # Key findings
    report.append("## Key Findings")
    report.append("")
    
    # Find best performer
    best_free = results["summary"]["nemotron"]["avg_heuristic_quality"]
    best_paid = max(results["summary"][m]["avg_heuristic_quality"] for m in ["phi4", "gpt4o_mini", "deepseek", "gpt4o"])
    
    report.append(f"1. **Free model quality:** Nemotron achieved {results['summary']['nemotron']['avg_heuristic_quality']}/10 heuristic quality")
    report.append(f"2. **Premium vs Free:** GPT-4o costs ${results['summary']['gpt4o']['total_cost']:.4f} vs Nemotron ${results['summary']['nemotron']['total_cost']:.4f}")
    
    if best_free >= best_paid * 0.9:
        report.append("3. **Free matches paid:** Free model within 10% of paid model quality")
    else:
        report.append(f"3. **Quality gap:** Free model at {best_free/best_paid*100:.0f}% of best paid")
    
    report.append("")
    report.append("## Verdict: Worth Publishing?")
    report.append("")
    
    # Calculate ROI
    if results["costs"]["total"] > 0:
        gpt4o_cost = results["summary"]["gpt4o"]["total_cost"]
        nemotron_cost = results["summary"]["nemotron"]["total_cost"]
        savings = (gpt4o_cost - nemotron_cost) / gpt4o_cost * 100 if gpt4o_cost > 0 else 0
        
        if savings > 70:
            verdict = "**YES** - Demonstrates significant cost savings (>70%)"
        elif savings > 50:
            verdict = "MAYBE - Moderate savings, needs more LLM-judged samples"
        else:
            verdict = "NO - Insufficient savings to justify publication"
        
        report.append(f"**Cost Savings:** {savings:.1f}% (Nemotron vs GPT-4o)")
        report.append(f"**Recommendation:** {verdict}")
    
    report.append("")
    report.append("*Report generated automatically by RouteSmith Scale-Up Experiment*")
    
    return "\n".join(report)

async def main():
    print("="*60)
    print("ROUTESMITH SCALE-UP EXPERIMENT")
    print(f"Queries: {len(QUERIES)}")
    print(f"Models: {len(MODELS)}")
    print(f"LLM-judged sample: {JUDGE_SAMPLE_SIZE}")
    print(f"Max budget: $30")
    print("="*60)
    
    # Run experiment
    results = await run_experiment()
    
    # Save raw results
    with open("scale_experiment_raw.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to scale_experiment_raw.json")
    
    # Generate report
    report = generate_report(results)
    with open("scale_experiment_results.md", "w") as f:
        f.write(report)
    print(f"Report saved to scale_experiment_results.md")
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total cost: ${results['costs']['total']:.4f}")
    print(f"{'='*60}")
    
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
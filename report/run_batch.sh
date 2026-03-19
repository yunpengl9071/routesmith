#!/bin/bash

# Define 15 customer support queries
QUERIES=(
  "How do I reset my password if I can't access my email?"
  "I was charged twice for my subscription. Can I get a refund?"
  "How do I upgrade my plan from Basic to Pro?"
  "I can't log into my account - it says my credentials are invalid"
  "Where can I find my billing history and invoices?"
  "How do I cancel my subscription?"
  "Your app keeps crashing when I try to upload a file"
  "I need to update my credit card information"
  "How do I enable two-factor authentication?"
  "I didn't receive the confirmation email for my order"
  "Can I get a refund within 30 days of purchase?"
  "How do I change my account username?"
  "Your service is down - I can't access anything"
  "How do I export my data from your platform?"
  "I want to delete my account permanently"
)

# Models to test
MODELS=("nvidia/nemotron-3-nano-30b-a3b:free" "microsoft/phi-4" "openai/gpt-4o-mini")

OUTPUT_FILE="/home/yliulupo/projects/routesmith/report/batch1_results.json"
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Initialize empty JSON array
echo "[]" > "$OUTPUT_FILE"

# Costs per 1M input/output tokens (approximate)
declare -A COSTS=(
  ["nvidia/nemotron-3-nano-30b-a3b:free"]=0
  ["microsoft/phi-4"]=0.0004
  ["openai/gpt-4o-mini"]=0.00015
)

# Run queries through each model
for query in "${QUERIES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Running: $model - ${query:0:50}..."
    
    # Call OpenRouter API
    response=$(curl -s -X POST "https://openrouter.ai/api/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $OPENROUTER_API_KEY" \
      -d "{
        \"model\": \"$model\",
        \"messages\": [{\"role\": \"user\", \"content\": \"You are a helpful customer support agent. Reply to this customer inquiry: $query\"}],
        \"max_tokens\": 200
      }")
    
    # Extract response content
    content=$(echo "$response" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
    
    # Get token usage for cost calculation
    input_tokens=$(echo "$response" | jq -r '.usage.prompt_tokens // 0' 2>/dev/null)
    output_tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)
    total_tokens=$((input_tokens + output_tokens))
    
    # Calculate cost (assuming ~$0 for free model, using input/output rates)
    cost=0
    if [[ "$model" == "openai/gpt-4o-mini" ]]; then
      cost=$(echo "scale=6; ($input_tokens * 0.00000015 + $output_tokens * 0.0000006) / 1000" | bc)
    elif [[ "$model" == "microsoft/phi-4" ]]; then
      cost=$(echo "scale=6; ($input_tokens * 0.0000002 + $output_tokens * 0.0000002) / 1000" | bc)
    fi
    
    # Escape content for JSON
    escaped_content=$(echo "$content" | jq -Rs .)
    escaped_query=$(echo "$query" | jq -Rs .)
    escaped_model=$(echo "$model" | jq -Rs .)
    
    # Add to JSON array
    tmp=$(mktemp)
    jq --arg q "$escaped_query" --arg m "$escaped_model" --arg r "$escaped_content" --argjson c "$cost" \
      '. += [{"query": $q, "model": $m, "response": $r, "cost": $c}]' \
      "$OUTPUT_FILE" > "$tmp" && mv "$tmp" "$OUTPUT_FILE"
    
    echo "  Cost: \$$cost"
  done
done

echo "Done! Results saved to $OUTPUT_FILE"
cat "$OUTPUT_FILE" | jq length

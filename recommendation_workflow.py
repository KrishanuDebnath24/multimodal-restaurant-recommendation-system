"""Multi-agent restaurant/recipe recommendation workflow.

Extracted from M3L2-Implement-Multi-Agent-Systems-v1.ipynb so the M3L3 chatbot
can call the same real workflow instead of a mock. The four phases are:

  1. User Analysis   (sequential) - build a profile from raw user data
  2. Data Retrieval  (sequential) - vector search over the M2L1 Chroma index
  3. Analysis        (parallel)   - trends, styles, nutrition
  4. Synthesis       (sequential) - final recommendations

Two entry points:
  run_workflow(user_input)             - all four phases, as in M3L2
  run_workflow_from_profile(profile)   - phases 2-4, for callers that already
                                         have a profile (e.g. the M3L3 chatbot,
                                         which extracts one from the chat message)

Prerequisites: run M2L1 to build ~/chroma_multimodal, and have Ollama serving
qwen2.5:7b at localhost:11434.
"""

import os
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

warnings.filterwarnings("ignore")

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# ================================
# LLM client
# ================================
# Point the OpenAI client at the local Ollama server instead of the hosted API.
# Ollama exposes an OpenAI-compatible endpoint at /v1, so the lab's
# client.chat.completions.create(...) code works unchanged.
# The api_key is required by the SDK but ignored by Ollama.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# The model to use (must already be pulled: `ollama pull qwen2.5:7b`)
MODEL = "qwen2.5:7b"

# ================================
# Connect to the Module 2 vector database
# ================================
# We talk to chromadb directly (the same engine langchain_chroma wraps) so this
# module does not depend on the langchain version installed for the agent code.

_HERE = Path(__file__).resolve().parent

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())
if not os.path.isdir(DB_DIR):
    raise RuntimeError(
        f"Vector database not found at '{DB_DIR}'. Run M2L1 to build it first."
    )

_chroma = chromadb.PersistentClient(path=DB_DIR)
article_col = _chroma.get_collection("restaurant_articles")
recipe_col = _chroma.get_collection("recipe_text")

# Full Module 1 records, keyed for enriching retrieval hits. Paths are anchored
# to this file so importing from a notebook works regardless of its cwd.
with open(_HERE / "structured_restaurant_data.json", "r", encoding="utf-8") as f:
    RESTAURANTS_RAW = json.load(f)
with open(_HERE / "augmented_food_recipe.json", "r", encoding="utf-8") as f:
    RECIPES_RAW = json.load(f)
RECIPES_BY_ID = {r["id"]: r for r in RECIPES_RAW}

# Locations present in the index, used to validate a filter before applying it.
KNOWN_LOCATIONS = {}
for _r in RESTAURANTS_RAW:
    _loc = str(_r.get("location", "")).strip()
    if _loc:
        KNOWN_LOCATIONS.setdefault(_loc.lower(), _loc)

# Query encoder - must match the model M2L1 used to build the text indexes.
# Both restaurant_articles and recipe_text are MiniLM (384-d), so one encoder
# serves both. The CLIP image collection is not needed by the agent pipeline.
text_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_query_text(query: str):
    """384-d MiniLM vector for searching restaurant_articles / recipe_text."""
    return text_model.encode([query], normalize_embeddings=True).astype(np.float32)[0]


print(f"✓ Vector DB connected — {article_col.count()} restaurants, "
      f"{recipe_col.count()} recipes, {len(KNOWN_LOCATIONS)} known locations")


# ================================
# Agent configurations from Lesson 1
# ================================
agent_configs = {
    "user_profile_generator": {
        "role": "User Profile Generator",
        "goal": "Analyze user restaurant visit history and social media posts to create a comprehensive profile.",
        "backstory": """You are an expert user behavior analyst with 10 years of experience in the food industry.
        You excel at identifying patterns in dining behavior and building rich user profiles."""
    },
    "rag_retriever": {
        "role": "RAG Retriever",
        "goal": "Query multimodal vector databases to retrieve relevant restaurants and recipes.",
        "backstory": """You are a data retrieval specialist with expertise in vector databases and semantic search."""
    },
    "food_trend_analyst": {
        "role": "Food Trend Analyst",
        "goal": "Identify current food trends and emerging dining concepts.",
        "backstory": """You are a culinary journalist who has spent 15 years covering food culture across global markets."""
    },
    "food_style_expert": {
        "role": "Food Style Expert",
        "goal": "Analyze cuisine types and flavor profiles to match user preferences.",
        "backstory": """You are a trained chef and culinary anthropologist with expertise in global cuisines."""
    },
    "nutrition_expert": {
        "role": "Nutrition Expert",
        "goal": "Evaluate nutritional content and ensure dietary compliance.",
        "backstory": """You are a registered dietitian with 8 years of clinical experience."""
    },
    "recommendation_expert": {
        "role": "Recommendation Expert",
        "goal": "Synthesize insights from all agents into final recommendations.",
        "backstory": """You are a recommendation systems architect with experience in personalization engines."""
    }
}


# ================================
# Shared state
# ================================
def initial_state(user_input: str = "") -> dict:
    """A fresh workflow state. Every node reads from and writes to this dict."""
    return {
        # Input
        "user_input": user_input,

        # Phase 1: User Analysis
        "user_profile": {},

        # Phase 2: Data Retrieval
        "retrieved_restaurants": [],
        "retrieved_recipes": [],

        # Phase 3: Analysis (Parallel)
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},

        # Phase 4: Synthesis
        "final_recommendations": {},

        # Metadata
        "workflow_step": "start",
    }


# ================================
# Helper: call an agent
# ================================
def call_agent(agent_key: str, user_message: str) -> str:
    """Call an agent with a specific message and return its response."""
    config = agent_configs[agent_key]

    system_prompt = f"""You are a {config['role']}.

Your goal: {config['goal']}

Your background: {config['backstory']}

Respond with structured, actionable output."""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


# ================================
# Node 1: Generate the user profile
# ================================
def node_generate_profile(state: dict) -> dict:
    """Generate user profile from input data."""
    print("\n[Phase 1] Generating user profile...")

    user_message = f"""Analyze this user data and create a comprehensive profile:

{state['user_input']}

Provide output in JSON format with these keys:
- favorite_cuisines (list)
- dietary_restrictions (list)
- dining_occasions (list)
- price_range (string)
- adventurousness_score (1-10)
- flavor_preferences (list)
- preferred_location (string: the neighbourhood or city the user dines in,
  copied verbatim from the input. Use "" if no location is mentioned —
  do not guess.)
- summary (string)
"""

    try:
        response = call_agent("user_profile_generator", user_message)
        print(f"""✓ User profile generated:
        {response}""")  # Debug line
        user_profile = json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        user_profile = {"error": f"Invalid JSON: {e}"}
    except Exception as e:
        print(f"⚠ Error generating profile: {e}")
        user_profile = {"error": str(e)}

    state["user_profile"] = user_profile
    state["workflow_step"] = "profile_generated"
    return state


# ================================
# Node 2: Retrieve the candidates
# ================================
def profile_to_query(profile: dict, fallback: str) -> str:
    """Flatten the Phase 1 user profile into a single search string."""
    if not isinstance(profile, dict) or profile.get("error"):
        return fallback[:500]  # profile generation failed - fall back to raw input

    bits = []
    for key in ("favorite_cuisines", "flavor_preferences",
                "dining_occasions", "dietary_restrictions", "preferred_location"):
        value = profile.get(key)
        if isinstance(value, list):
            bits += [str(v) for v in value]
        elif value:
            bits.append(str(value))
    if profile.get("summary"):
        bits.append(str(profile["summary"]))

    return ", ".join(bits) if bits else fallback[:500]


def location_filter(profile: dict):
    """Build a Chroma where-clause from the profile's preferred_location.

    Only locations that actually exist in the index are used - a filter on an
    unknown value would silently return zero hits.
    """
    if not isinstance(profile, dict):
        return None, ""
    raw = str(profile.get("preferred_location", "")).strip()
    if not raw:
        return None, ""
    canonical = KNOWN_LOCATIONS.get(raw.lower())
    if not canonical:
        return None, raw  # named, but not somewhere we have restaurants
    return {"location": canonical}, canonical


def _cosine(distance: float) -> float:
    """Chroma stores squared L2 over L2-normalised vectors, so cos = 1 - d/2."""
    return round(1.0 - float(distance) / 2.0, 4)


def node_retrieve_candidates(state: dict) -> dict:
    """Retrieve restaurant and recipe candidates from the Module 2 vector database.

    Unlike the other nodes this one makes no LLM call - retrieval is a
    deterministic vector search, so the candidates are real records from the
    index rather than plausible-sounding inventions.
    """
    print("\n[Phase 2] Retrieving candidates from vector database...")

    profile = state["user_profile"]
    query = profile_to_query(profile, state["user_input"])
    where, loc = location_filter(profile)

    print(f"  query: {query[:120]}")
    if where:
        print(f"  location filter: {loc}")
    elif loc:
        print(f"  location '{loc}' is not in the index — searching all locations")

    restaurants, recipes = [], []
    try:
        # ---- restaurants: text -> text over restaurant_articles (MiniLM, 384-d) ----
        res = article_col.query(
            query_embeddings=[embed_query_text(query).tolist()],
            n_results=10,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        if where and not res["ids"][0]:
            print(f"  no restaurants in {loc} — retrying without the filter")
            res = article_col.query(
                query_embeddings=[embed_query_text(query).tolist()],
                n_results=10,
                include=["documents", "metadatas", "distances"],
            )

        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            idx = int(str(meta.get("doc_id", "rest_-1")).split("_")[-1])
            src = RESTAURANTS_RAW[idx] if 0 <= idx < len(RESTAURANTS_RAW) else {}
            restaurants.append({
                "name": src.get("name"),
                "cuisine": src.get("food_style"),
                "location": src.get("location"),
                "type": src.get("type"),
                "rating": src.get("rating"),
                "price_range": src.get("price_range"),
                "signatures": src.get("signatures", []),
                # M2L1 embeds both Vibe and Environment into the indexed document,
                # so they already steer retrieval - carry them through as well so
                # the analysis agents can reason about atmosphere explicitly and
                # the chatbot can show it.
                "vibe": src.get("vibe"),
                "environment": src.get("environment"),
                "similarity": _cosine(dist),
            })

        # ---- recipes: text -> text over recipe_text (MiniLM, 384-d) ----
        # Searches names, cuisines, ingredients and the M1L2 captions, so
        # ingredient- and diet-aware queries work (CLIP image search cannot
        # tell that paneer is dairy).
        res = recipe_col.query(
            query_embeddings=[embed_query_text(query).tolist()],
            n_results=20,
            include=["documents", "metadatas", "distances"],
        )
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            src = RECIPES_BY_ID.get(meta.get("recipe_id"), {})
            recipes.append({
                "name": src.get("name"),
                "cuisine": src.get("cuisine"),
                "prep_time": src.get("prep_time"),
                "total_time": src.get("total_time"),
                "servings": src.get("servings"),
                "ingredients": src.get("ingredients", []),
                # image caption generated by the vision model in M1L2
                "description": src.get("image_description", ""),
                "similarity": _cosine(dist),
            })

        print(f"✓ Retrieved {len(restaurants)} restaurants and {len(recipes)} recipes")
        for r in restaurants[:3]:
            print(f"    [{r['similarity']:.3f}] {r['name']} — {r['cuisine']} ({r['location']})")
        for r in recipes[:3]:
            print(f"    [{r['similarity']:.3f}] {r['name']} — {r['cuisine']}")
    except Exception as e:
        print(f"⚠ Error retrieving candidates: {e}")
        restaurants, recipes = [], []

    state["retrieved_restaurants"] = restaurants
    state["retrieved_recipes"] = recipes
    state["workflow_step"] = "candidates_retrieved"
    return state


# ================================
# Node 3: Analyze trends
# ================================
def node_analyze_trends(state: dict) -> dict:
    """Analyze food trends in the retrieved candidates."""
    print("\n[Phase 3a] Analyzing food trends...")

    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]

    user_message = f"""Analyze current food trends in these options:

Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Identify 3-5 relevant trends and explain how they align with modern dining culture.
Return JSON: {{"trends": [{{"name": str, "description": str, "relevance": str}}]}}
"""

    try:
        response = call_agent("food_trend_analyst", user_message)
        trend_analysis = json.loads(response)
        print(f"✓ Identified {len(trend_analysis.get('trends', []))} trends")
    except Exception as e:
        print(f"⚠ Error analyzing trends: {e}")
        trend_analysis = {"error": str(e)}

    state["trend_analysis"] = trend_analysis
    return state


# ================================
# Node 4: Analyze food styles
# ================================
def node_analyze_styles(state: dict) -> dict:
    """Analyze food styles and flavor profiles."""
    print("\n[Phase 3b] Analyzing food styles...")

    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]

    user_message = f"""Analyze the cuisine types and flavor profiles of these restaurants and recipes:

User Profile: {json.dumps(profile, indent=2)}

Restaurants: {json.dumps(restaurants[:5], indent=2)}

Recipes: {json.dumps(recipes[:5], indent=2)}

For each item, identify:
- Cuisine classification (regional, cultural, fusion)
- Dominant flavor profiles (umami-rich, bright/acidic, rich/creamy, spicy, delicate, etc.)
- Key cooking methods and techniques
- Ingredient signatures
- Alignment with user preferences

Return JSON: {{"cuisine_analysis": [{{"item_name": str, "item_type": str, "cuisine": str, "flavor_profiles": [str], "techniques": [str], "alignment_score": float, "match_reasoning": str}}], "summary": str}}"""

    try:
        response = call_agent("food_style_expert", user_message)
        style_analysis = json.loads(response)
        print("✓ Style analysis completed")
    except Exception as e:
        print(f"⚠ Error analyzing styles: {e}")
        style_analysis = {"error": str(e)}

    state["style_analysis"] = style_analysis
    return state


# ================================
# Node 5: Evaluate nutrition
# ================================
def node_evaluate_nutrition(state: dict) -> dict:
    """Evaluate nutritional aspects and dietary compliance."""
    print("\n[Phase 3c] Evaluating nutrition...")

    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]

    user_message = f"""Evaluate the nutritional fit of these options:

User Profile: {json.dumps(profile, indent=2)}
Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Check dietary restrictions, allergens, and nutritional balance.
Return JSON: {{"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}}
"""

    try:
        response = call_agent("nutrition_expert", user_message)
        nutrition_analysis = json.loads(response)
        print("✓ Nutrition evaluation completed")
    except Exception as e:
        print(f"⚠ Error evaluating nutrition: {e}")
        nutrition_analysis = {"error": str(e)}

    state["nutrition_analysis"] = nutrition_analysis
    return state


# ================================
# Node 6: Generate recommendations
# ================================
def node_generate_recommendations(state: dict) -> dict:
    """Synthesize all analyses into final recommendations."""
    print("\n[Phase 4] Generating final recommendations...")

    user_message = f"""Synthesize these insights into top 5 restaurant and top 5 recipe recommendations:

User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants: {json.dumps(state['retrieved_restaurants'][:10], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:10], indent=2)}
Trends: {json.dumps(state['trend_analysis'], indent=2)}
Styles: {json.dumps(state['style_analysis'], indent=2)}
Nutrition: {json.dumps(state['nutrition_analysis'], indent=2)}

Only recommend items that appear in the Restaurants and Recipes lists above,
and copy their names verbatim.

Return JSON:
{{
  "restaurants": [{{"name": str, "reasoning": str}}],
  "recipes": [{{"name": str, "reasoning": str}}]
}}

Each reasoning should be 2-3 sentences explaining why it's a great match.
"""

    try:
        response = call_agent("recommendation_expert", user_message)
        recommendations = json.loads(response)
        print(f"✓ Generated {len(recommendations.get('restaurants', []))} restaurant recommendations")
        print(f"✓ Generated {len(recommendations.get('recipes', []))} recipe recommendations")
    except Exception as e:
        print(f"⚠ Error generating recommendations: {e}")
        recommendations = {"error": str(e)}

    state["final_recommendations"] = recommendations
    state["workflow_step"] = "complete"
    return state


# ================================
# Workflow graph
# ================================
def _run_analysis_and_synthesis(state: dict) -> dict:
    """Phases 3 and 4: parallel analysis, then sequential synthesis."""
    print("\n[Phase 3] Running analysis agents in parallel...")

    # Each function needs its own copy of state to read from,
    # and we merge their outputs back afterwards.
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_trends = executor.submit(node_analyze_trends, dict(state))
        future_styles = executor.submit(node_analyze_styles, dict(state))
        future_nutrition = executor.submit(node_evaluate_nutrition, dict(state))

        result_trends = future_trends.result()
        result_styles = future_styles.result()
        result_nutrition = future_nutrition.result()

    # Merge parallel results back into state
    state["trend_analysis"] = result_trends["trend_analysis"]
    state["style_analysis"] = result_styles["style_analysis"]
    state["nutrition_analysis"] = result_nutrition["nutrition_analysis"]

    # Phase 4 – Sequential
    return node_generate_recommendations(state)


def run_workflow(user_input: str) -> dict:
    """Run the full multi-agent workflow.

    Phases:
      1. User Analysis      (sequential)
      2. Data Retrieval     (sequential)
      3. Analysis           (parallel – trends, styles, nutrition)
      4. Synthesis          (sequential)
    """
    state = initial_state(user_input)

    # Phase 1 – Sequential
    state = node_generate_profile(state)

    # Phase 2 – Sequential
    state = node_retrieve_candidates(state)

    # Phases 3 & 4
    return _run_analysis_and_synthesis(state)


def run_workflow_from_profile(profile: dict, user_input: str = "") -> dict:
    """Run phases 2-4 against a profile the caller already has.

    Used by the M3L3 chatbot, which builds a profile from the chat message with
    its own extract_preferences() - re-deriving it in Phase 1 would spend an
    extra LLM call on work already done. `user_input` is only a fallback search
    string for profile_to_query when the profile carries nothing usable.
    """
    state = initial_state(user_input)
    state["user_profile"] = profile or {}
    state["workflow_step"] = "profile_generated"

    # Phase 2 – Sequential
    state = node_retrieve_candidates(state)

    # Phases 3 & 4
    return _run_analysis_and_synthesis(state)


# ================================
# Evaluation
# ================================
def evaluate_recommendations(result: Dict[str, Any]):
    """Evaluate the quality of recommendations."""
    print("\n" + "=" * 80)
    print("RECOMMENDATION EVALUATION")
    print("=" * 80)

    profile = result.get("user_profile", {})
    recommendations = result.get("final_recommendations", {})

    # Check if recommendations exist
    restaurants = recommendations.get("restaurants", [])
    recipes = recommendations.get("recipes", [])

    print(f"\n✓ Number of restaurant recommendations: {len(restaurants)}")
    print(f"✓ Number of recipe recommendations: {len(recipes)}")

    # Check dietary compliance
    dietary_restrictions = profile.get("dietary_restrictions", [])
    if dietary_restrictions:
        print(f"\n✓ Dietary restrictions identified: {', '.join(dietary_restrictions)}")
        print("  → Check if recommendations respect these restrictions")

    # Check diversity
    favorite_cuisines = profile.get("favorite_cuisines", [])
    if favorite_cuisines:
        print(f"\n✓ Favorite cuisines: {', '.join(favorite_cuisines)}")
        print("  → Check if recommendations include these cuisines")

    # Evaluate reasoning quality
    if restaurants:
        print("\n✓ First restaurant recommendation:")
        print(f"  Name: {restaurants[0].get('name', 'N/A')}")
        print(f"  Reasoning: {restaurants[0].get('reasoning', 'N/A')}")

    if recipes:
        print("\n✓ First recipe recommendation:")
        print(f"  Name: {recipes[0].get('name', 'N/A')}")
        print(f"  Reasoning: {recipes[0].get('reasoning', 'N/A')}")

    print("\n" + "=" * 80)

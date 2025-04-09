# Playground

import pandas as pd
import Levenshtein
from typing import Dict, Any, List, Optional, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain.tools import Tool
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer, util
import torch
import json

# Define a typed dict for our state
class MatchingState(TypedDict, total=False):
    """Dictionary type for attribute matching state"""
    original_attribute: str
    reference_data: List[Dict[str, Any]]
    levenshtein_matches: List[tuple]
    embedding_matches: List[tuple]
    llm_judgment: str
    messages: List[Dict[str, str]]

# Initialize LLM
llm = ChatOllama(model="llama3.2:1b", temperature=0)

# Load embedding model
def get_embedding_model():
    """Initialize the embedding model (only once)."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Function to calculate Levenshtein distance and get top matches
def calculate_levenshtein(state: Dict) -> Dict:
    """
    Calculate Levenshtein distance for the attribute and get top 3 matches.
    
    Args:
        state: Current state with original_attribute and reference_data
    
    Returns:
        Updated state with Levenshtein matches
    """
    # Create a new state to avoid modifying the input
    new_state = state.copy()
    
    # Get the attribute name and reference data
    name = state["original_attribute"]
    reference_data = state["reference_data"]
    
    # Convert reference data to DataFrame if it's not already
    if not isinstance(reference_data, pd.DataFrame):
        df = pd.DataFrame(reference_data)
    else:
        df = reference_data
    
    # Filter valid match names
    match_names = df[df['match name'].notna()].copy()
    
    # Calculate Levenshtein ratio for each match name
    match_names['distance'] = match_names['match name'].apply(lambda x: Levenshtein.ratio(name, x))
    
    # Get the top 3 matches sorted by distance in descending order
    top_matches = match_names.nlargest(3, 'distance')
    
    # Create a list of tuples (match name, distance formatted to 4 decimal places)
    top_matches_list = [(row['match name'], float(f"{row['distance']:.4f}")) for _, row in top_matches.iterrows()]
    
    # Store the results
    new_state["levenshtein_matches"] = top_matches_list
    
    # Check if the first match has a low score
    first_match_score = top_matches_list[0][1] if top_matches_list else 0
    new_state["needs_embedding"] = first_match_score < 0.9
    
    # Add message
    if "messages" not in new_state:
        new_state["messages"] = []
    
    matches_str = ", ".join([f"({m[0]}, {m[1]:.4f})" for m in top_matches_list])
    new_state["messages"].append({"role": "assistant", "content": f"Levenshtein matches: {matches_str}"})
    
    if new_state["needs_embedding"]:
        new_state["messages"].append({"role": "assistant", "content": f"First match score {first_match_score:.4f} < 0.9, using embeddings next"})
    else:
        new_state["messages"].append({"role": "assistant", "content": f"First match score {first_match_score:.4f} >= 0.9, skipping embeddings"})
    
    return new_state

# Function to calculate embeddings similarity if needed
def calculate_embeddings(state: Dict) -> Dict:
    """
    Calculate embedding similarity for the attribute.
    
    Args:
        state: Current state with original_attribute and reference_data
    
    Returns:
        Updated state with embedding matches
    """
    # Create a new state to avoid modifying the input
    new_state = state.copy()
    
    # If we don't need embeddings, return unchanged state
    if not state.get("needs_embedding", False):
        new_state["messages"].append({"role": "assistant", "content": "Skipping embeddings calculation"})
        return new_state
    
    # Get the attribute name and reference data
    name = state["original_attribute"]
    reference_data = state["reference_data"]
    
    # Convert reference data to DataFrame if it's not already
    if not isinstance(reference_data, pd.DataFrame):
        df = pd.DataFrame(reference_data)
    else:
        df = reference_data
    
    # Filter valid match names
    valid_df = df[df['match name'].notna()]
    
    # Get the embedding model
    model = get_embedding_model()
    
    # Encode the current name and all reference names
    query_embedding = model.encode([name], convert_to_tensor=True)
    df_embeddings = model.encode(valid_df['match name'].tolist(), convert_to_tensor=True)
    
    # Calculate cosine similarity
    cosine_scores = util.pytorch_cos_sim(query_embedding, df_embeddings)[0]
    
    # Get the top 3 match indices
    top_match_indices = torch.topk(cosine_scores, k=min(3, len(cosine_scores))).indices.tolist()
    
    # Retrieve the top 3 match names and their similarity scores
    top_3_matches = [
        (valid_df['match name'].iloc[idx], round(cosine_scores[idx].item(), 4))
        for idx in top_match_indices
    ]
    
    # Store the results
    new_state["embedding_matches"] = top_3_matches
    
    # Add message
    matches_str = ", ".join([f"({m[0]}, {m[1]:.4f})" for m in top_3_matches])
    new_state["messages"].append({"role": "assistant", "content": f"Embedding matches: {matches_str}"})
    
    return new_state

# Function to use LLM to judge the best match
def judge_matches_with_llm(state: Dict) -> Dict:
    """
    Use LLM to judge the best matches between Levenshtein and embedding results.
    
    Args:
        state: Current state with matches and reference data
    
    Returns:
        Updated state with LLM judgment
    """
    # Create a new state to avoid modifying the input
    new_state = state.copy()
    
    # Get the attribute name and matches
    original_attr = state["original_attribute"]
    levenshtein_matches = state.get("levenshtein_matches", [])
    embedding_matches = state.get("embedding_matches", [])
    
    # Get reference data
    reference_data = state["reference_data"]
    
    # Convert reference data to DataFrame if it's not already
    if not isinstance(reference_data, pd.DataFrame):
        df = pd.DataFrame(reference_data)
    else:
        df = reference_data
    
    # Create a dictionary to store attribute-definition pairs
    attr_definitions = {}
    
    # Add Levenshtein matches to the dictionary
    for match, _ in levenshtein_matches:
        definition = df[df['match name'] == match]['definition'].iloc[0] if match in df['match name'].values else "No definition"
        attr_definitions[match] = definition
    
    # Add embedding matches to the dictionary if they're not already included
    for match, _ in embedding_matches:
        if match not in attr_definitions:
            definition = df[df['match name'] == match]['definition'].iloc[0] if match in df['match name'].values else "No definition"
            attr_definitions[match] = definition
    
    # Format the attribute-definition pairs for the prompt
    attributes_with_defs = "\n".join([f"{attr}: {definition}" for attr, definition in attr_definitions.items()])
    
    # Create the prompt for the LLM
    prompt = f"""
    Original attribute: "{original_attr}"
    
    Available attribute choices with definitions:
    {attributes_with_defs}
    
    Levenshtein top matches: {[(m[0], float(f"{m[1]:.4f}")) for m in levenshtein_matches]}
    Embedding top matches: {[(m[0], float(f"{m[1]:.4f}")) for m in embedding_matches] if embedding_matches else "Not calculated"}
    
    Select which attribute from the choices above is the closest match to the original attribute "{original_attr}".
    Return only the name of the best matching attribute, with no other text.
    """
    
    # Call the LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Store the LLM's judgment
    new_state["llm_judgment"] = response.content.strip()
    
    # Add message
    new_state["messages"].append({"role": "assistant", "content": f"LLM selected best match: {new_state['llm_judgment']}"})
    
    return new_state

# Create the tools
calculate_levenshtein_tool = Tool(
    name="calculate_levenshtein",
    func=calculate_levenshtein,
    description="Calculate Levenshtein distance and get top matches"
)

calculate_embeddings_tool = Tool(
    name="calculate_embeddings",
    func=calculate_embeddings,
    description="Calculate embedding similarity for low-confidence matches"
)

judge_matches_tool = Tool(
    name="judge_matches",
    func=judge_matches_with_llm,
    description="Use LLM to judge the best match"
)

# Define the workflow edges
def needs_embedding(state: Dict) -> str:
    """Check if embedding calculation is needed based on Levenshtein results."""
    if state.get("needs_embedding", False):
        return "needs_embedding"
    else:
        return "skip_embedding"

# Build the graph for processing a single row
def build_attribute_matching_graph():
    # Create the graph
    workflow = StateGraph(state_schema=Dict)
    
    # Add nodes
    workflow.add_node("calculate_levenshtein", calculate_levenshtein_tool)
    workflow.add_node("calculate_embeddings", calculate_embeddings_tool)
    workflow.add_node("judge_matches", judge_matches_tool)
    
    # Add edges
    workflow.set_entry_point("calculate_levenshtein")
    
    # Conditional branch after calculate_levenshtein
    workflow.add_conditional_edges(
        "calculate_levenshtein",
        needs_embedding,
        {
            "needs_embedding": "calculate_embeddings",
            "skip_embedding": "judge_matches"
        }
    )
    
    workflow.add_edge("calculate_embeddings", "judge_matches")
    workflow.add_edge("judge_matches", END)
    
    # Compile the graph
    return workflow.compile()

# Function to process a single attribute
def process_attribute(attribute, reference_data):
    """
    Process a single attribute through the matching pipeline.
    
    Args:
        attribute: The attribute name to match
        reference_data: DataFrame or list of dictionaries with reference data
        
    Returns:
        Dictionary with matches and best match
    """
    # Build the graph
    graph = build_attribute_matching_graph()
    
    # Set up the initial state
    initial_state = {
        "original_attribute": attribute,
        "reference_data": reference_data,
        "messages": []
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return result

# Function to process a DataFrame of attributes
def process_dataframe(input_df, reference_df):
    """
    Process each row in the input DataFrame.
    
    Args:
        input_df: DataFrame with attributes to match
        reference_df: DataFrame with reference attributes and definitions
        
    Returns:
        DataFrame with matches and best matches for each attribute
    """
    # Create a copy of the input DataFrame to store results
    result_df = input_df.copy()
    
    # Add columns for results if they don't exist
    if "levenshtein_top_3" not in result_df.columns:
        result_df["levenshtein_top_3"] = None
    if "embedding_top_3" not in result_df.columns:
        result_df["embedding_top_3"] = None
    if "best_match" not in result_df.columns:
        result_df["best_match"] = None
    
    # Process each row
    for i, row in input_df.iterrows():
        print(f"Processing row {i+1}/{len(input_df)}: {row['match name']}")
        
        # Process the attribute
        result = process_attribute(row["match name"], reference_df)
        
        # Store the results
        result_df.at[i, "levenshtein_top_3"] = str(result.get("levenshtein_matches", []))
        result_df.at[i, "embedding_top_3"] = str(result.get("embedding_matches", []))
        result_df.at[i, "best_match"] = result.get("llm_judgment", "")
        
        # Print messages
        for msg in result.get("messages", []):
            print(f"[{msg.get('role', 'unknown')}] {msg.get('content', '')}")
    
    return result_df

# Main function to run the workflow
def main():
    # Load dataframes
    input_df = pd.read_csv("result.csv")
    reference_df = pd.read_csv("reference_attributes.csv")
    
    # Process all rows
    result_df = process_dataframe(input_df, reference_df)
    
    # Save the result dataframe to a CSV file
    output_path = "attribute_matching_results.csv"
    result_df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()

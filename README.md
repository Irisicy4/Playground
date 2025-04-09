import pandas as pd
import Levenshtein
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize LLM
llm = ChatOllama(model="llama3.2:1b", temperature=0)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define our process functions - each takes and returns a state dictionary
def calculate_levenshtein(state: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Levenshtein distance and get top 3 matches"""
    # Get data from state
    name = state["attribute"]
    reference_df = state["reference_df"]
    
    # Filter valid match names
    match_names = reference_df[reference_df['match name'].notna()].copy()
    
    # Calculate Levenshtein ratio for each match name
    match_names['distance'] = match_names['match name'].apply(lambda x: Levenshtein.ratio(name, x))
    
    # Get the top 3 matches sorted by distance in descending order
    top_matches = match_names.nlargest(3, 'distance')
    
    # Create a list of tuples (match name, distance formatted to 4 decimal places)
    top_matches_list = [(row['match name'], float(f"{row['distance']:.4f}")) for _, row in top_matches.iterrows()]
    
    # Update state with results
    state["levenshtein_matches"] = top_matches_list
    
    # Check if the first match has a low score
    first_match_score = top_matches_list[0][1] if top_matches_list else 0
    state["needs_embedding"] = first_match_score < 0.9
    
    print(f"Levenshtein matches for '{name}':")
    for match, score in top_matches_list:
        print(f"  - {match}: {score:.4f}")
    
    return state

def calculate_embeddings(state: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate embedding similarity for low-confidence matches"""
    # Get data from state
    name = state["attribute"]
    reference_df = state["reference_df"]
    
    # Filter valid match names
    valid_df = reference_df[reference_df['match name'].notna()]
    
    # Encode the current name and all reference names
    query_embedding = embedding_model.encode([name], convert_to_tensor=True)
    df_embeddings = embedding_model.encode(valid_df['match name'].tolist(), convert_to_tensor=True)
    
    # Calculate cosine similarity
    cosine_scores = util.pytorch_cos_sim(query_embedding, df_embeddings)[0]
    
    # Get the top 3 match indices
    top_match_indices = torch.topk(cosine_scores, k=min(3, len(cosine_scores))).indices.tolist()
    
    # Retrieve the top 3 match names and their similarity scores
    top_3_matches = [
        (valid_df['match name'].iloc[idx], round(cosine_scores[idx].item(), 4))
        for idx in top_match_indices
    ]
    
    # Update state with results
    state["embedding_matches"] = top_3_matches
    
    print(f"Embedding matches for '{name}':")
    for match, score in top_3_matches:
        print(f"  - {match}: {score:.4f}")
    
    return state

def judge_matches_with_llm(state: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to judge the best match"""
    # Get data from state
    name = state["attribute"]
    reference_df = state["reference_df"]
    levenshtein_matches = state["levenshtein_matches"]
    embedding_matches = state.get("embedding_matches", [])
    
    # Create a dictionary to store attribute-definition pairs
    attr_definitions = {}
    
    # Add Levenshtein matches to the dictionary
    for match, _ in levenshtein_matches:
        definition = reference_df[reference_df['match name'] == match]['definition'].iloc[0] if match in reference_df['match name'].values else "No definition"
        attr_definitions[match] = definition
    
    # Add embedding matches to the dictionary if they're not already included
    for match, _ in embedding_matches:
        if match not in attr_definitions:
            definition = reference_df[reference_df['match name'] == match]['definition'].iloc[0] if match in reference_df['match name'].values else "No definition"
            attr_definitions[match] = definition
    
    # Format the attribute-definition pairs for the prompt
    attributes_with_defs = "\n".join([f"{attr}: {definition}" for attr, definition in attr_definitions.items()])
    
    # Create the prompt for the LLM
    prompt = f"""
    Original attribute: "{name}"
    
    Available attribute choices with definitions:
    {attributes_with_defs}
    
    Levenshtein top matches: {[(m[0], float(f"{m[1]:.4f}")) for m in levenshtein_matches]}
    Embedding top matches: {[(m[0], float(f"{m[1]:.4f}")) for m in embedding_matches] if embedding_matches else "Not calculated"}
    
    Select which attribute from the choices above is the closest match to the original attribute "{name}".
    Return only the name of the best matching attribute, with no other text.
    """
    
    # Call the LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Update state with results
    state["llm_judgment"] = response.content.strip()
    
    print(f"LLM judgment for '{name}': {state['llm_judgment']}")
    
    return state

# Condition function to determine if we need embeddings
def needs_embedding(state: Dict[str, Any]) -> str:
    """Check if we need to calculate embeddings based on Levenshtein results"""
    if state.get("needs_embedding", False):
        return "needs_embedding"
    else:
        return "skip_embedding"

# Build the graph
def build_attribute_matching_graph():
    # Create the workflow
    workflow = StateGraph(state_schema=Dict)
    
    # Add nodes
    workflow.add_node("calculate_levenshtein", calculate_levenshtein)
    workflow.add_node("calculate_embeddings", calculate_embeddings)
    workflow.add_node("judge_matches", judge_matches_with_llm)
    
    # Set the entry point
    workflow.set_entry_point("calculate_levenshtein")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "calculate_levenshtein",
        needs_embedding,
        {
            "needs_embedding": "calculate_embeddings",
            "skip_embedding": "judge_matches"
        }
    )
    
    # Add remaining edges
    workflow.add_edge("calculate_embeddings", "judge_matches")
    workflow.add_edge("judge_matches", END)
    
    # Compile the graph
    return workflow.compile()

# Function to process a single attribute
def process_single_attribute(attribute: str, reference_df: pd.DataFrame) -> Dict[str, Any]:
    """Process a single attribute through the graph"""
    # Build the graph
    graph = build_attribute_matching_graph()
    
    # Initialize the state
    initial_state = {
        "attribute": attribute,
        "reference_df": reference_df
    }
    
    # Process the attribute
    result = graph.invoke(initial_state)
    
    return result

# Main function to process the CSV
def process_csv(input_csv: str, reference_csv: str, output_csv: str = "attribute_matching_results.csv"):
    """Process each row in the input CSV"""
    # Load the CSV files
    input_df = pd.read_csv(input_csv)
    reference_df = pd.read_csv(reference_csv)
    
    # Create columns for results if they don't exist
    if "levenshtein_top_3" not in input_df.columns:
        input_df["levenshtein_top_3"] = None
    if "embedding_top_3" not in input_df.columns:
        input_df["embedding_top_3"] = None
    if "best_match" not in input_df.columns:
        input_df["best_match"] = None
    
    # Process each row
    for i, row in input_df.iterrows():
        attribute = row["match name"]
        print(f"\nProcessing row {i+1}/{len(input_df)}: {attribute}")
        
        # Process the attribute
        result = process_single_attribute(attribute, reference_df)
        
        # Store the results
        input_df.at[i, "levenshtein_top_3"] = str(result["levenshtein_matches"])
        input_df.at[i, "embedding_top_3"] = str(result.get("embedding_matches", []))
        input_df.at[i, "best_match"] = result["llm_judgment"]
    
    # Save the results
    input_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    process_csv("result.csv", "reference_attributes.csv")

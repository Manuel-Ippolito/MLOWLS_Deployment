import pandas as pd


def primary_id_to_common_name(id: int | str) -> str:
    taxonomy_df = pd.read_csv("data/taxonomy.csv")

    # Convert id to string and normalize
    id_str = str(id).strip().lower()
    
    # Find the row with matching primary_label (convert to string and strip whitespace)
    result = taxonomy_df[taxonomy_df['primary_label'].astype(str).str.strip().str.lower() == id_str]

    if result.empty:
        raise ValueError(f"No entry found for primary_label: {id}")

    return result['common_name'].iloc[0]

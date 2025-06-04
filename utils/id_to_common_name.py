import pandas as pd


def primary_id_to_common_name(id: int) -> str:
    taxonomy_df = pd.read_csv("data/taxonomy.csv")
    
    # Find the row with matching primary_label
    result = taxonomy_df[taxonomy_df['primary_label'] == str(id)]

    if result.empty:
        raise ValueError(f"No entry found for primary_label: {id}")

    return result['common_name'].iloc[0]

import os
import pandas as pd

def convert_csv_to_data_and_info(csv_file_path,data_files_path):
    # Load CSV file, removing header if it exists
    df = pd.read_csv(csv_file_path, header=0)
    
    # # Remove FILEID column if present
    # if 'FILEID' in df.columns[0] or df.columns[0].lower() == 'fileid' or df.columns[0].lower() == 'FileID':
    #     df = df.iloc[:, 1:]
        
    # Remove polarity column if present in the last position
    if df.columns[-1].lower() in ['polarity', 'positive', 'negative']:
        df = df.iloc[:, :-1]
    
    # Generate the output .data file path
    data_file_name = os.path.basename(csv_file_path).replace('.csv', '.data')
    data_file_path = os.path.join(data_files_path,data_file_name)
    # Format all feature columns to 3 decimal places, ensure label is integer (0 or 1)
    formatted_df = df.copy()
    for col in df.columns[1:-1]:  # Apply formatting to all except the label column
        formatted_df[col] = df[col].map(lambda x: '%.3f' % x if pd.notnull(x) else '')
    
    # Ensure the label column is integer without floating point
    formatted_df.iloc[:, -1] = df.iloc[:, -1].astype(int)
    
    # Save the formatted DataFrame to a .data file
    formatted_df.to_csv(data_file_path, index=False, header=False, sep=',', float_format='%.3f')
    print(f"Converted {csv_file_path} to {data_file_path}")
    
    # Generate the .info file
    num_features = len(df.columns) - 1  # Subtract label column
    info_file_name = os.path.basename(csv_file_path).replace('.csv', '.INFO')
    info_file_path = os.path.join(data_files_path,info_file_name)
    with open(info_file_path, "w") as file:
        # Write continuous types for all features
        for i in range(1, num_features + 1):
            file.write(f"{i} continuous\n")
        # Write the label type as discrete
        file.write("class discrete\n")
        file.write("LABEL_POS -1\n")
    
    print(f"{info_file_path} has been created successfully.")


if __name__ == '__main__':
    # Example usage
    csv_file = "/mnt/d/embeddings/Partialspoof_specnet_emb.csv"
    data_files_path = "/mnt/d/rrl_ssl/dataset"
    convert_csv_to_data_and_info(csv_file,data_files_path)

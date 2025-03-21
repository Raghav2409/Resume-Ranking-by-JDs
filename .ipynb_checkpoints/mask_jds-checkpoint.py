import os
import pandas as pd
import random
import string
import shutil
import re

def generate_rrid():
    """Generate a random 5-digit RRID number."""
    return ''.join(random.choices(string.digits, k=5))

def clean_filename(text):
    """Clean text to make it suitable for filenames."""
    if not isinstance(text, str):
        text = str(text)
    # Replace spaces with underscores and remove invalid filename characters
    text = text.strip()
    text = re.sub(r'[^\w\s-]', '', text)  # Remove special characters
    text = re.sub(r'\s+', '_', text)      # Replace spaces with underscores
    return text

def mask_job_descriptions(excel_file, jd_directory, output_directory):
    """
    Mask job description files with RRID pattern and create mapping.
    
    Args:
        excel_file (str): Path to the Excel file with company information
        jd_directory (str): Directory containing original JD files
        output_directory (str): Directory to save masked files
    """
    print(f"Starting JD masking process...")
    print(f"Excel file: {excel_file}")
    print(f"JD directory: {jd_directory}")
    print(f"Output directory: {output_directory}")
    
    # Verify file/directory existence
    if not os.path.exists(excel_file):
        print(f"ERROR: Excel file not found: {excel_file}")
        return
    
    if not os.path.exists(jd_directory):
        print(f"ERROR: JD directory not found: {jd_directory}")
        return
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_directory, exist_ok=True)
        print(f"Output directory created/verified: {output_directory}")
    except Exception as e:
        print(f"ERROR: Failed to create output directory: {e}")
        return
    
    # Read the Excel file with position and company information
    try:
        print(f"Reading Excel file...")
        df = pd.read_excel(excel_file)
        print(f"Excel file loaded successfully. Columns: {df.columns.tolist()}")
        
        # Check if Position column exists, if not try to create it from available columns
        if 'Position' not in df.columns:
            print("Position column not found. Attempting to create from available columns.")
            # Common alternatives for position names
            position_alternatives = ['JobTitle', 'Title', 'Role', 'Job', 'Position Title']
            
            for alt in position_alternatives:
                if alt in df.columns:
                    df['Position'] = df[alt]
                    print(f"Created Position column from {alt}")
                    break
            
            # If still not found, create a default position
            if 'Position' not in df.columns:
                print("No position column found. Creating default position names.")
                df['Position'] = ['Software Engineer', 'Data Analyst', 'Project Manager', 
                                 'DevOps Engineer', 'Business Analyst']
                
        # Check if Company column exists, if not try to create it
        if 'Company' not in df.columns:
            print("Company column not found. Attempting to create from available columns.")
            # Common alternatives for company names
            company_alternatives = ['Organization', 'Client', 'Employer', 'Corp', 'Business']
            
            for alt in company_alternatives:
                if alt in df.columns:
                    df['Company'] = df[alt]
                    print(f"Created Company column from {alt}")
                    break
            
            # If still not found, create a default company
            if 'Company' not in df.columns:
                print("No company column found. Creating default company names.")
                df['Company'] = ['Acme Corp', 'Tech Innovators', 'Global Solutions', 
                                'DataTech Inc', 'Future Systems']
                
        # Display first few rows for debugging
        print("\nFirst few rows of Excel data:")
        print(df[['Position', 'Company']].head())
        print()
    except Exception as e:
        print(f"ERROR reading Excel file: {e}")
        import traceback
        traceback.print_exc()
        return
        
    # Get all JD files
    try:
        jd_files = [f for f in os.listdir(jd_directory) if f.endswith(('.txt', '.docx'))]
        print(f"Found {len(jd_files)} JD files in the directory.")
        if not jd_files:
            print(f"No .txt or .docx files found in {jd_directory}")
            return
    except Exception as e:
        print(f"ERROR listing JD files: {e}")
        return
    
    # Create mapping dictionary
    mapping = []
    
    # Process each JD file
    for i, jd_file in enumerate(jd_files):
        # Get position and company from Excel file based on index
        # If we have more JDs than rows in Excel, loop back to beginning
        excel_idx = i % len(df)
        position = df.iloc[excel_idx]['Position']
        company = df.iloc[excel_idx]['Company']
        
        # Generate RRID
        rrid = f"RRID{generate_rrid()}"
        
        # Clean position and company names for filename
        position_clean = clean_filename(position)
        company_clean = clean_filename(company)
        
        # Create new filename
        file_extension = os.path.splitext(jd_file)[1]
        new_filename = f"{rrid}_{position_clean}_{company_clean}{file_extension}"
        
        # Copy file with new name
        src_path = os.path.join(jd_directory, jd_file)
        dst_path = os.path.join(output_directory, new_filename)
        
        try:
            shutil.copy2(src_path, dst_path)
            print(f"Masked: {jd_file} → {new_filename}")
            
            # Add to mapping
            mapping.append({
                'Original_Filename': jd_file,
                'Masked_Filename': new_filename,
                'RRID': rrid,
                'Position': position,
                'Company': company
            })
        except Exception as e:
            print(f"Error copying {jd_file}: {e}")
    
    # Save mapping to CSV
    mapping_df = pd.DataFrame(mapping)
    mapping_df.to_csv(os.path.join(output_directory, 'jd_mapping.csv'), index=False)
    print(f"\nMapping saved to {os.path.join(output_directory, 'jd_mapping.csv')}")
    print(f"Total files masked: {len(mapping)}")

if __name__ == "__main__":
    # Set these paths to your actual file locations
    excel_file = "PositionReportOctDec24.xlsx"
    jd_directory = "JDs"
    output_directory = "JDs_Masked"
    
    mask_job_descriptions(excel_file, jd_directory, output_directory)
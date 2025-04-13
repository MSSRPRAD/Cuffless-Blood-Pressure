import duckdb
import os
from pathlib import Path

def process_info_file(info_file_path, output_dir, is_supplementary=False):
    # Create output directory if it doesn't exist
    output_base = Path(output_dir)
    if is_supplementary:
        output_base = output_base / "Supplementary"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Get output filename
    output_filename = Path(info_file_path).stem.replace("_Info", "_Processed") + ".parquet"
    output_path = output_base / output_filename
    
    # Initialize DuckDB connection
    con = duckdb.connect()
    
    # Query with COPY command
    query = f"""
    SET enable_progress_bar = true;
    SET memory_limit = '200GB';
    COPY (
        WITH TargetRows AS (
            SELECT 
                S.SubjectID,
                S.SegmentID,
                S.CaseID,
                CASE 
                    WHEN S.filename LIKE '%MIMIC%' THEN 'MIMIC'
                    WHEN S.filename LIKE '%Vital%' THEN 'VitalDB'
                END AS Source,
                ROW_NUMBER() OVER (PARTITION BY S.SubjectID, 
                                    Source
                                  ORDER BY S.SegmentID) as RowNum
            FROM read_parquet('./PulseDB_*/*parquet', union_by_name = true, filename = true) S
        ),
        ValidSegments AS (
            SELECT 
                B.name as SubjectID,
                B.Subj_SegIDX as RequiredRowNum,
                B.Source as Source
            FROM read_parquet('{info_file_path}') B
        ),
        MatchingSegments AS (
            SELECT 
                T.SubjectID,
                T.SegmentID,
                T.CaseID,
                T.Source
            FROM TargetRows T
            INNER JOIN ValidSegments V
                ON T.SubjectID = V.SubjectID 
                AND T.RowNum = V.RequiredRowNum
                AND T.Source = V.Source
        )
        SELECT D.*,
        CASE
            WHEN D.filename LIKE '%MIMIC%' THEN 'MIMIC'
            WHEN D.filename LIKE '%Vital%' THEN 'VitalDB'
            ELSE 'Unknown'
        END AS Source
        FROM read_parquet('./PulseDB_*/*parquet', union_by_name = true, filename = true) D
        INNER JOIN MatchingSegments M
            ON D.SubjectID = M.SubjectID 
            AND D.SegmentID = M.SegmentID
            AND D.CaseID = M.CaseID
            AND CASE
                WHEN D.filename LIKE '%MIMIC%' THEN 'MIMIC'
                WHEN D.filename LIKE '%Vital%' THEN 'VitalDB'
                ELSE 'Unknown'
            END = M.Source
    ) TO '{output_path}' (FORMAT 'parquet', COMPRESSION 'SNAPPY');
    """
    
    # Execute query
    try:
        print(f"Processing {info_file_path}...")
        con.execute(query)
        print(f"Saved results to {output_path}")
    except Exception as e:
        print(f"Error processing {info_file_path}: {str(e)}")
    finally:
        con.close()

def main():
    # Create base output directory
    output_dir = "dataset"
    
    # Process main Info_Files
    info_files_dir = "Info_Files"
    for file in os.listdir(info_files_dir):
        if file.endswith(".parquet"):
            process_info_file(
                os.path.join(info_files_dir, file),
                output_dir,
                is_supplementary=False
            )
    
    # Process Supplementary_Info_Files
    supp_info_files_dir = "Supplementary_Info_Files"
    for file in os.listdir(supp_info_files_dir):
        if file.endswith(".parquet"):
            process_info_file(
                os.path.join(supp_info_files_dir, file),
                output_dir,
                is_supplementary=True
            )

if __name__ == "__main__":
    main()

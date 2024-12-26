import polars as pl
import numpy as np
from pathlib import Path


def get_subset(info_file_path: str, low_memory=True, columns=None): 
    # Load all the segment files into a lazy frame
    # Load the info file
    parquet_files = Path('./Segment_Files').rglob('*.parquet')
    parquet_files = [str(pf) for pf in parquet_files]
    df = pl.scan_parquet(parquet_files, low_memory=low_memory).lazy()

    if columns:
        df = df.select(columns)
    
    df_info = pl.scan_parquet(info_file_path, low_memory=low_memory)
   
    # Make the join
    df_info = df_info.select(['Subj_SegIDX', 'name'])
    
    result = df.join(df_info, right_on=['name', 'Subj_SegIDX'], left_on=['SubjectID', 'SegmentID'], how='inner')
    
    return result

name = "AAMI_Cal_Info.parquet"
subset = get_subset("./Info_Files/AAMI_Cal_Info.parquet")
subset.sink_parquet("subsets/")

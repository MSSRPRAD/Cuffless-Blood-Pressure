import numpy as np
import h5py
import os
from pathlib import Path
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
import sys
from tqdm.auto import tqdm
import gc
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import polars as pl
from joblib import Parallel, delayed

def sid_asstr(sid):
    numbers = sid.flatten().tolist()
    return ''.join(chr(num) for num in numbers)

def segidx_as_float(seg_idx):
    return seg_idx[0][0]

def prepare_parquet_from_infofile_inplace(path: str):
    try:
        with h5py.File(path, 'r') as f:
            filename = Path(path).stem
            print(f"Processing {filename}")
            print("Keys in file:", list(f.keys()))
            key = None
            for k in list(f.keys()):
                if k != "#refs#":
                    key = k
            subj_seg_indices = [segidx_as_float(np.array(f[it])) for it in f[key]['Subj_SegIDX'][0]]
            sids_raw = [sid_asstr(np.array(f[it])) for it in f[key]['Subj_Name'][0]]
            source = [sid_asstr(np.array(f[it])) for it in f[key]['Source'][0]]
            print(type(subj_seg_indices[0]), type(sids_raw[0]))
            print(subj_seg_indices[0], sids_raw[0])
            print(len(subj_seg_indices), len(sids_raw)) 
            print("Reached Here!")
            df = pl.DataFrame({
                'combined': sids_raw,
                'Subj_SegIDX': subj_seg_indices,
                'Source': source
            })
            df = df.with_columns([
              pl.col('combined').str.split('_').list.get(0).alias('name'),
              pl.col('combined').str.split('_').list.get(1).alias('flag')])
            df = df.drop('combined')
            df.write_parquet(path.split('.')[0] + '.parquet')
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
    return []

def prepare_parquet_indir(directory: str):
    # Ensure the directory exists
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist")
    
    # Get all .mat files in the directory
    mat_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                 if f.endswith('.mat')]
    
    if not mat_files:
        print(f"No .mat files found in {directory}")
        return
    
    # Process each file
    for mat_file in mat_files:
        print(f"\nProcessing file: {mat_file}")
        prepare_parquet_from_infofile_inplace(mat_file)

def df_from_segfile(path):
    """
    Extracts rows from a single .mat file and returns them as a Polars DataFrame.
    """
    feature_shape = {}
    rows_data = []
    try:
        with h5py.File(path, 'r') as f:
            features = list(f['Subj_Wins'].keys())
            rows = f['Subj_Wins'][features[0]].shape[1]
            print(f"Rows: {rows}")
            for row in range(rows):
                row_data = {}
                for feature in features:
                    # Extracting data for the current feature and row
                    if f['Subj_Wins'][feature].shape != (1, rows):
                        print("ERROR!!!!!! SCHEMA WRONG")
                        return None
                    data = np.array(f[f['Subj_Wins'][feature][0][row]])
                   
                    if row==0:
                        if feature == features[0]:
                            print(f"Num of Features: {len(features)}")
                        print(f"Row {row}, Feature {feature}, Shape: {data.shape}")
                    #if data.shape == (1, 1):
                    #    data = data[0][0]
                    #elif data.shape[0] == 1:
                    #    data = data[0].tolist()
                    #else:
                    #    data = sid_asstr(data)
                    
                    if data.shape[0] == 1:
                        data = data[0].tolist()
                    else:
                        data = sid_asstr(data)

                    # Check shape consistency. Print otherwise
                    if feature not in feature_shape.keys():
                        feature_shape[feature] = np.array(data).shape
                    
                    if feature not in ['ABP_Turns', 'ABP_SPeaks', 'PPG_Turns', 'ECG_RPeaks', 'PPG_SPeaks']:
                        if(np.array(data).shape != feature_shape[feature]):
                            print(f"Shape conflict for feature: {feature}!!!!")
                            print(f"Previously Found: {feature_shape[feature].shape} | Now Found {data.shape}")
 
                    row_data[feature] = data
                rows_data.append(row_data)
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
    df = pl.DataFrame(rows_data, strict=False)
    for col in df.columns:
        if col in ['ABP_SPeaks', 'ABP_Turns', 'ECG_RPeaks', 'PPG_SPeaks', 'PPG_Turns']:
            continue
        if str(type(df[col].dtype)) == 'List':
            df = df.with_columns(
            pl.col(col).list.to_array(len(df[col][0])).alias(col)
        )
    return df


import tempfile

def process_single_file(file_path):
    try:
        output_file = file_path.replace('.mat', '.parquet')
        lock_file = output_file + '.lock'

        # Attempt to create a lock file atomically
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)

        if os.path.exists(output_file):
            print(f"Skipping {file_path}, corresponding Parquet file already exists.")
            os.remove(lock_file)
            return

        print(f"Processing file: {file_path}")
        df = df_from_segfile(file_path)
        if (df is not None) and (df.shape[0]*df.shape[1] != 0):
            df.write_parquet(output_file)
            print(f"Successfully wrote Parquet file: {output_file}")
        else:
            print(f"Empty or invalid dataframe from: {file_path}")

    except FileExistsError:
        # Another process is handling this file
        print(f"Skipping {file_path}, lock already exists.")
        return
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
    finally:
        # Clean up lock file
        if os.path.exists(lock_file):
            os.remove(lock_file)
        gc.collect()


def create_parquet_from_segmentfiles_parallel(directory: str):
    """
    Processes all .mat files in the given directory and writes their data
    to individual Parquet files.

    Uses parallel processing to handle files efficiently.
    """
    mat_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mat')]

    if not mat_files:
        print(f"No .mat files found in {directory}")
        return

    def signal_handler(sig, frame):
        print("\nTermination requested. Stopping all processes...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_single_file, file): file for file in mat_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                future.result()
            except Exception as e:
                print(f"Error during parallel processing: {e}")
import os
import sys
import signal
import polars as pl
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import gc
from pathlib import Path

def process_parquet(file_path: str) -> str:
    """Load a parquet file in chunks, print its shape, clear memory."""
    try:
        changed = False
        df = pl.read_parquet(file_path)
        columns2check = ['ABP_Lag', 'Age', 'BMI', 'Gender', 'Height', 'IncludeFlag', 'PPG_ABP_Corr', 'SegDBP', 'SegSBP', 'SegmentID', 'Weight', 'WinID', 'WinSeqID']
        for i in range(len(df.columns)):
            col = df.columns[i]
            if col in columns2check and str(df.schema[col]).startswith("Array"):
                df = df.with_columns(
                        pl.col(col).arr.first().alias(col)
                    )
                changed = True
        if changed:
            df.write_parquet(file_path)
        return ""
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return str(e)

def visit_parquets(folder_name: str, n_jobs: int = 100):
    """Visit and process parquet files in the folder using Joblib."""
    # Collect all parquet files
    parquet_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_name)
        for file in files if file.endswith(".parquet")
    ]
    
    with tqdm(total=len(parquet_files), desc="Processing files") as pbar:
        try:
            # Process files in parallel using Joblib
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(process_parquet)(file_path) 
                for file_path in tqdm(parquet_files, leave=False)
            )
        except KeyboardInterrupt:
            print("\nTerminating all processes...")
            sys.exit(1)

# Signal handler
def handle_sigint(signum, frame):
    print("\nSIGINT received, shutting down...")
    sys.exit(1)

signal.signal(signal.SIGINT, handle_sigint)


if __name__ == "__main__":
    prepare_parquet_indir("Supplementary_Info_Files")
    prepare_parquet_indir("Info_Files")
    #create_parquet_from_segmentfiles_parallel("./PulseDB_MIMIC")
    #create_parquet_from_segmentfiles_parallel("./PulseDB_Vital")
    #process_single_file("Segment_Files/PulseDB_MIMIC/p059845.mat")
    #process_single_file("Segment_Files/PulseDB_MIMIC/p068919.mat")
    #visit_parquets("Segment_Files")
    #visit_parquets("./Supplementary_Info_Files")
    #visit_parquets("./Info_Files")
    #process_parquet("./Segment_Files/PulseDB_Vital/p000110.parquet")

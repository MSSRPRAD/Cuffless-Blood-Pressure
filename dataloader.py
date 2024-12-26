import duckdb
from typing import Iterator, List, Dict
import os
from pprint import pprint as pp

def cols2sql(columns):
    columns = ['D.' + col for col in columns]
    return ','.join(columns)

class DataLoader:
    def __init__(self, info_file_path: str, segment_files_path: str, batch_size: int = 32):
        """
        Initialize the DataLoader with file paths and batch size.
        
        Args:
            info_file_path: Path to the Info_Files/Train_Info.parquet
            segment_files_path: Path to the directory containing segment parquet files
            batch_size: Number of rows to return in each batch
        """
        self.info_file_path = info_file_path
        self.segment_files_path = segment_files_path
        self.batch_size = batch_size
        self.con = duckdb.connect(":memory:")
        self.offset = 0  # Initialize offset to 0
        
        # Get total number of rows
        self.total_rows = self._get_total_rows()
        
    def _get_total_rows(self) -> int:
        """Get the total number of valid rows from the info file."""
        query = f"""
        SELECT COUNT(*) as count 
        FROM read_parquet('{self.info_file_path}')
        """
        return self.con.execute(query).fetchone()[0]
    
    def get_batches(self, columns: List[str]):
        """
        Generate batches of data using DuckDB's OFFSET and LIMIT.
        
        Yields:
            List of dictionaries containing the batch data
        """
        query = f"""
        WITH TargetRows AS (
      SELECT 
          S.SubjectID,
          S.SegmentID,
          S.CaseID,
          ROW_NUMBER() OVER (PARTITION BY S.SubjectID ORDER BY S.SegmentID) as RowNum
      FROM read_parquet('./Segment_Files/*/*parquet') S
  ),
  ValidSegments AS (
      SELECT 
          B.name as SubjectID,
          B.Subj_SegIDX as RequiredRowNum
      FROM read_parquet('./Info_Files/Train_Info.parquet') B 
      LIMIT {self.batch_size} OFFSET {self.offset}
  ),
  MatchingSegments AS (
      SELECT 
          T.SubjectID,
          T.SegmentID,
          T.CaseID
      FROM TargetRows T
      INNER JOIN ValidSegments V
          ON T.SubjectID = V.SubjectID 
          AND T.RowNum = V.RequiredRowNum
  )
  SELECT 
      {cols2sql(columns)}
  FROM read_parquet('./Segment_Files/*/*parquet') D
  INNER JOIN MatchingSegments M
      ON D.SubjectID = M.SubjectID 
      AND D.SegmentID = M.SegmentID
      AND D.CaseID = M.CaseID;
        """
        
        result = self.con.execute(query).fetchall()
        if not result:
            return
        # Update offset for next batch
        self.offset += self.batch_size
        # Convert to list of dictionaries
        batch = [dict(zip(columns, row)) for row in result]
        yield batch
    
    def __len__(self) -> int:
        """Return the total number of rows."""
        return self.total_rows

import polars as pl
if __name__ == '__main__':
    # Initialize the loader
    loader = DataLoader(
        info_file_path="./Info_Files/Train_Info.parquet",
        segment_files_path="./Segment_Files",
        batch_size=1024
    )
    
    # Print total number of rows
    print(f"Total rows: {len(loader)}")
    
    # Accessing the batch directly
    for i in range(10):
        batch_generator = loader.get_batches(columns=['SubjectID', 'SegmentID', 'SegSBP', 'SegDBP', 'PPG_F', 'ABP_F', 'Age', 'Gender'])
        print("===========================")
        try:
            batch = next(batch_generator)
            print(f"Batch {i + 1}:")
            print(f"Type of batch: {type(batch)}")
            print(f"Batch contents:")
            df = pl.DataFrame(batch)
            print(df.shape)
            print(df.head())
            print("===============================")
            input()
        except StopIteration:
            print("No more batches available.")
            break


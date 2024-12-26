import duckdb
from typing import Iterator, List, Dict, Optional
import os

class DataLoader:
    def __init__(self, info_file_path: str, segment_files_path: str, batch_size: int = 32, random_seed: Optional[int] = None, random_sort: bool = False):
        self.info_file_path = info_file_path
        self.segment_files_path = segment_files_path
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.random_sort = random_sort
        self.con = duckdb.connect(":memory:")
        self.total_rows = self._get_total_rows()
        
    def _get_total_rows(self) -> int:
        query = f"""
        SELECT COUNT(*) as count 
        FROM read_parquet('{self.info_file_path}')
        """
        return self.con.execute(query).fetchone()[0]
    
    def get_batches(self, columns: Optional[List[str]] = None) -> Iterator[List[Dict]]:
        select_clause = '*' if columns is None else ', '.join(columns)
        
        random_seed_clause = f"select setseed({self.random_seed});" if self.random_seed is not None else ""
        order_clause = "ORDER BY random()" if self.random_sort else ""
        
        for offset in range(0, self.total_rows, self.batch_size):
            query = f"""
            {random_seed_clause}
            WITH RankedSegments AS (
                SELECT 
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY A.SubjectID 
                        ORDER BY A.SegmentID
                    ) as RowNum
                FROM read_parquet('{self.segment_files_path}/*/*.parquet') A
            ),
            ValidSegments AS (
                SELECT 
                    B.name as SubjectID,
                    B.Subj_SegIDX as RequiredRowNum
                FROM read_parquet('{self.info_file_path}') B
            ),
            FilteredSegments AS (
                SELECT 
                    *,
                    ROW_NUMBER() OVER (ORDER BY random()) as batch_row_num
                FROM RankedSegments R
                INNER JOIN ValidSegments V
                    ON R.SubjectID = V.SubjectID 
                    AND R.RowNum = V.RequiredRowNum
            )
            SELECT {select_clause}
            FROM FilteredSegments
            {order_clause}
            OFFSET {offset}
            LIMIT {self.batch_size};
            """
            
            result = self.con.execute(query).fetchall()
            if not result:
                break
                
            column_names = [desc[0] for desc in self.con.description] if columns is None else columns
            batch = [dict(zip(column_names, row)) for row in result]
            yield batch
            
    def __len__(self) -> int:
        return self.total_rows


if __name__ == "__main__":
    loader = DataLoader(
        info_file_path="./Info_Files/Train_Info.parquet",
        segment_files_path="./Segment_Files",
        batch_size=3,
        random_seed=0.42,
        random_sort=True
    )
    
    print(f"Total rows in dataset: {len(loader)}")
    
    print("\nTest 1: Getting all columns")
    print("-" * 50)
    for i, batch in enumerate(loader.get_batches()):
        print(f"Batch {i + 1}:")
        for row in batch:
            print(row)
        if i >= 1:
            break
            
    print("\nTest 2: Getting specific columns")
    print("-" * 50)
    selected_columns = ['SubjectID', 'SegSBP']
    for i, batch in enumerate(loader.get_batches(columns=selected_columns)):
        print(f"Batch {i + 1}:")
        for row in batch:
            print(row)
        if i >= 1:
            break
            
    print("\nTest 3: Testing with different random seed")
    print("-" * 50)
    loader_new = DataLoader(
        info_file_path="./Info_Files/Train_Info.parquet",
        segment_files_path="./Segment_Files",
        batch_size=3,
        random_seed=100,
        random_sort=True
    )
    for i, batch in enumerate(loader_new.get_batches()):
        print(f"Batch {i + 1}:")
        for row in batch:
            print(row)
        if i >= 0:
            break

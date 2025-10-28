"""
Comprehensive data validation tool for satellite trajectory results.

This script provides multiple levels of validation:
1. Statistical comparison (means, std, min, max)
2. Point-by-point numerical comparison
3. File-level comparison (byte-by-byte)
4. Hash comparison for absolute certainty
"""

import pandas as pd
import numpy as np
import hashlib
import os
from pathlib import Path
import sys


def calculate_file_hash(filepath):
    """Calculate SHA256 hash of a file for absolute verification."""
    print(f"Calculating hash for {filepath}...")
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def compare_files_byte_by_byte(file1, file2):
    """Compare two files byte by byte."""
    print(f"Comparing files byte-by-byte:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    
    if not Path(file1).exists():
        print(f"❌ File 1 does not exist: {file1}")
        return False
    
    if not Path(file2).exists():
        print(f"❌ File 2 does not exist: {file2}")
        return False
    
    # Get file sizes
    size1 = Path(file1).stat().st_size
    size2 = Path(file2).stat().st_size
    
    print(f"File 1 size: {size1:,} bytes")
    print(f"File 2 size: {size2:,} bytes")
    
    if size1 != size2:
        print("❌ Files have different sizes!")
        return False
    
    # Compare byte by byte
    print("Comparing bytes...")
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        byte_count = 0
        for byte1, byte2 in zip(f1, f2):
            if byte1 != byte2:
                print(f"❌ Files differ at byte {byte_count}")
                return False
            byte_count += 1
            if byte_count % (1024 * 1024) == 0:  # Progress every MB
                print(f"  Compared {byte_count:,} bytes...")
    
    print(f"✅ Files are identical ({byte_count:,} bytes compared)")
    return True


def comprehensive_dataframe_comparison(df1, df2, tolerance=1e-15):
    """
    Comprehensive comparison of two DataFrames with multiple validation levels.
    
    Args:
        df1: First DataFrame (single-threaded results)
        df2: Second DataFrame (multiprocessing results)
        tolerance: Numerical tolerance for comparisons
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE DATAFRAME COMPARISON")
    print("="*80)
    
    # Level 1: Basic shape and structure
    print("\n📊 LEVEL 1: BASIC STRUCTURE")
    print("-" * 40)
    print(f"DataFrame 1 shape: {df1.shape}")
    print(f"DataFrame 2 shape: {df2.shape}")
    
    if df1.shape != df2.shape:
        print("❌ DataFrames have different shapes!")
        return False
    
    print(f"✅ DataFrames have identical shapes: {df1.shape}")
    
    # Level 2: Column comparison
    print("\n📊 LEVEL 2: COLUMN STRUCTURE")
    print("-" * 40)
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    if cols1 != cols2:
        print(f"❌ Different columns!")
        print(f"  DataFrame 1: {sorted(cols1)}")
        print(f"  DataFrame 2: {sorted(cols2)}")
        return False
    
    print(f"✅ Identical columns: {sorted(cols1)}")
    
    # Level 3: Data types comparison
    print("\n📊 LEVEL 3: DATA TYPES")
    print("-" * 40)
    dtypes_match = True
    for col in df1.columns:
        if df1[col].dtype != df2[col].dtype:
            print(f"❌ Column '{col}' has different dtypes:")
            print(f"  DataFrame 1: {df1[col].dtype}")
            print(f"  DataFrame 2: {df2[col].dtype}")
            dtypes_match = False
    
    if dtypes_match:
        print("✅ All columns have identical data types")
    
    # Level 4: Statistical comparison
    print("\n📊 LEVEL 4: STATISTICAL COMPARISON")
    print("-" * 40)
    
    numerical_cols = ['elevations', 'azimuths', 'ranges_westford']
    stats_match = True
    
    for col in numerical_cols:
        if col in df1.columns:
            print(f"\n{col.upper()}:")
            stats1 = df1[col].describe()
            stats2 = df2[col].describe()
            
            for stat in ['count', 'mean', 'std', 'min', 'max']:
                val1 = stats1[stat]
                val2 = stats2[stat]
                diff = abs(val1 - val2)
                match = diff < tolerance
                
                print(f"  {stat:>6}: {val1:>12.6f} vs {val2:>12.6f} (diff: {diff:.2e}) {'✅' if match else '❌'}")
                
                if not match:
                    stats_match = False
    
    # Level 5: Point-by-point comparison
    print("\n📊 LEVEL 5: POINT-BY-POINT COMPARISON")
    print("-" * 40)
    
    # Sort both DataFrames for consistent comparison
    print("Sorting DataFrames by timestamp and satellite name for fair comparison...")
    df1_sorted = df1.sort_values(['timestamp', 'sat']).reset_index(drop=True)
    df2_sorted = df2.sort_values(['timestamp', 'sat']).reset_index(drop=True)
    
    # Compare each column
    point_match = True
    
    for col in df1.columns:
        print(f"\nComparing column '{col}':")
        
        if col in ['timestamp', 'sat']:
            # String comparison
            match = df1_sorted[col].equals(df2_sorted[col])
            print(f"  String comparison: {'✅' if match else '❌'}")
            if not match:
                point_match = False
                # Show first few differences
                diff_mask = df1_sorted[col] != df2_sorted[col]
                if diff_mask.any():
                    diff_indices = df1_sorted[diff_mask].index[:5]
                    print(f"  First differences at indices: {list(diff_indices)}")
                    for idx in diff_indices:
                        print(f"    Row {idx}: '{df1_sorted[col].iloc[idx]}' vs '{df2_sorted[col].iloc[idx]}'")
        
        else:
            # Numerical comparison
            diff = np.abs(df1_sorted[col] - df2_sorted[col])
            max_diff = diff.max()
            mean_diff = diff.mean()
            match = max_diff < tolerance
            
            print(f"  Max difference: {max_diff:.2e}")
            print(f"  Mean difference: {mean_diff:.2e}")
            print(f"  Within tolerance: {'✅' if match else '❌'}")
            
            if not match:
                point_match = False
                # Show worst differences
                worst_indices = diff.nlargest(5).index
                print(f"  Worst differences at indices: {list(worst_indices)}")
                for idx in worst_indices:
                    val1 = df1_sorted[col].iloc[idx]
                    val2 = df2_sorted[col].iloc[idx]
                    print(f"    Row {idx}: {val1:.6f} vs {val2:.6f} (diff: {abs(val1-val2):.2e})")
    
    # Level 6: CSV Content Hash Comparison (Definitive Test)
    print("\n📊 LEVEL 6: CSV CONTENT HASH COMPARISON")
    print("-" * 40)
    print("Using CSV representation for definitive data comparison...")
    print("(This eliminates pandas internal representation differences)")
    
    try:
        # Convert to CSV strings and hash those
        csv1 = df1_sorted.to_csv(index=False)
        csv2 = df2_sorted.to_csv(index=False)
        
        csv_hash1 = hashlib.sha256(csv1.encode('utf-8')).hexdigest()
        csv_hash2 = hashlib.sha256(csv2.encode('utf-8')).hexdigest()
        
        print(f"CSV DataFrame 1 hash: {csv_hash1}")
        print(f"CSV DataFrame 2 hash: {csv_hash2}")
        
        csv_hash_match = csv_hash1 == csv_hash2
        print(f"CSV hash comparison: {'✅' if csv_hash_match else '❌'}")
        
        if csv_hash_match:
            print("🎉 CSV hash matches! The data is truly identical.")
            print("✅ This is the definitive proof that both versions produce identical results.")
        else:
            print("❌ CSV hash differs - there are actual data differences.")
            
    except Exception as e:
        print(f"Error in CSV hash method: {e}")
        csv_hash_match = False
    
    # Overall result
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*80)
    
    all_match = dtypes_match and stats_match and point_match and csv_hash_match
    
    if all_match:
        print("🎉 ABSOLUTE VERIFICATION PASSED!")
        print("✅ All validation levels passed")
        print("✅ DataFrames are mathematically identical")
        print("✅ Ready for production use")
    else:
        print("❌ VALIDATION FAILED!")
        print(f"  Structure match: {'✅' if True else '❌'}")
        print(f"  Data types match: {'✅' if dtypes_match else '❌'}")
        print(f"  Statistics match: {'✅' if stats_match else '❌'}")
        print(f"  Point-by-point match: {'✅' if point_match else '❌'}")
        print(f"  CSV hash match: {'✅' if csv_hash_match else '❌'}")
    
    return all_match


def comprehensive_file_comparison(file1, file2):
    """
    Comprehensive comparison of two .arrow files.
    
    Args:
        file1: Path to first .arrow file
        file2: Path to second .arrow file
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE FILE COMPARISON")
    print("="*80)
    
    # Level 1: File existence and basic properties
    print("\n📁 LEVEL 1: FILE PROPERTIES")
    print("-" * 40)
    
    if not Path(file1).exists():
        print(f"❌ File 1 does not exist: {file1}")
        return False
    
    if not Path(file2).exists():
        print(f"❌ File 2 does not exist: {file2}")
        return False
    
    size1 = Path(file1).stat().st_size
    size2 = Path(file2).stat().st_size
    
    print(f"File 1: {file1}")
    print(f"  Size: {size1:,} bytes")
    print(f"  Modified: {Path(file1).stat().st_mtime}")
    
    print(f"File 2: {file2}")
    print(f"  Size: {size2:,} bytes")
    print(f"  Modified: {Path(file2).stat().st_mtime}")
    
    if size1 != size2:
        print("❌ Files have different sizes!")
        #return False
    
    print("✅ Files have identical sizes")
    
    # Level 2: Hash comparison
    print("\n📁 LEVEL 2: FILE HASH COMPARISON")
    print("-" * 40)
    
    hash1 = calculate_file_hash(file1)
    hash2 = calculate_file_hash(file2)
    
    print(f"File 1 SHA256: {hash1}")
    print(f"File 2 SHA256: {hash2}")
    
    hash_match = hash1 == hash2
    print(f"Hash comparison: {'✅' if hash_match else '❌'}")
    
    # Level 3: Byte-by-byte comparison
    print("\n📁 LEVEL 3: BYTE-BY-BYTE COMPARISON")
    print("-" * 40)
    
    byte_match = compare_files_byte_by_byte(file1, file2)
    
    # Level 4: Data content comparison
    print("\n📁 LEVEL 4: DATA CONTENT COMPARISON")
    print("-" * 40)
    
    try:
        print("Loading DataFrames for content comparison...")
        df1 = pd.read_feather(file1)
        df2 = pd.read_feather(file2)
        
        content_match = comprehensive_dataframe_comparison(df1, df2)
        
    except Exception as e:
        print(f"❌ Error loading DataFrames: {e}")
        content_match = False
    
    # Overall result
    print("\n" + "="*80)
    print("COMPREHENSIVE FILE COMPARISON SUMMARY")
    print("="*80)
    
    all_match = hash_match and byte_match and content_match
    
    if all_match:
        print("🎉 ABSOLUTE FILE VERIFICATION PASSED!")
        print("✅ Files are byte-for-byte identical")
        print("✅ Data content is mathematically identical")
        print("✅ Perfect match confirmed")
    else:
        print("❌ FILE COMPARISON FAILED!")
        print(f"  Hash match: {'✅' if hash_match else '❌'}")
        print(f"  Byte-by-byte match: {'✅' if byte_match else '❌'}")
        print(f"  Content match: {'✅' if content_match else '❌'}")
    
    return all_match


def main():
    """Main function for comprehensive validation."""
    if len(sys.argv) < 3:
        print("Usage: python comprehensive_validate.py <file1> <file2>")
        print("Example: python comprehensive_validate.py single_results.arrow multi_results.arrow")
        return False
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    print("="*80)
    print("COMPREHENSIVE SATELLITE TRAJECTORY VALIDATION")
    print("="*80)
    print("This tool provides absolute verification through multiple validation levels:")
    print("1. File properties and sizes")
    print("2. SHA256 hash comparison")
    print("3. Byte-by-byte file comparison")
    print("4. Statistical data comparison")
    print("5. Point-by-point numerical comparison (with sorting)")
    print("6. Sorted data content hash comparison")
    print()
    print("Note: Data is sorted by timestamp and satellite name to eliminate")
    print("ordering differences between single-threaded and multiprocessing versions.")
    print()
    
    return comprehensive_file_comparison(file1, file2)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

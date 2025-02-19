import pandas as pd
import numpy as np
import os
from datetime import datetime

def calculate_poa(ra2c_file, ra2cfull_file):
    """Calculate Price of Anarchy by comparing RA2C and RA2CFull performance"""
    # Read the CSV files
    ra2c_df = pd.read_csv(ra2c_file)
    ra2cfull_df = pd.read_csv(ra2cfull_file)
    
    # Calculate metrics for both approaches
    metrics = {}
    
    # 1. Average Reward
    metrics['ra2c_reward'] = ra2c_df['reward'].mean()
    metrics['ra2cfull_reward'] = ra2cfull_df['reward'].mean()
    
    # 2. Average Energy Consumption
    # First, get the action columns
    ra2c_action_cols = [col for col in ra2c_df.columns if 'action_' in col]
    ra2cfull_action_cols = [col for col in ra2cfull_df.columns if 'action_' in col]
    
    metrics['ra2c_energy'] = ra2c_df[ra2c_action_cols].mean().mean()
    metrics['ra2cfull_energy'] = ra2cfull_df[ra2cfull_action_cols].mean().mean()
    
    # 3. Average AoI
    ra2c_aoi_cols = [col for col in ra2c_df.columns if 'aoi_' in col]
    ra2cfull_aoi_cols = [col for col in ra2cfull_df.columns if 'aoi_' in col]
    
    metrics['ra2c_aoi'] = ra2c_df[ra2c_aoi_cols].mean().mean()
    metrics['ra2cfull_aoi'] = ra2cfull_df[ra2cfull_aoi_cols].mean().mean()
    
    # Calculate PoA for each metric
    # Note: For reward, higher is better, so we invert the ratio
    poa = {
        'reward_poa': metrics['ra2cfull_reward'] / metrics['ra2c_reward'],
        'energy_poa': metrics['ra2c_energy'] / metrics['ra2cfull_energy'],
        'aoi_poa': metrics['ra2c_aoi'] / metrics['ra2cfull_aoi']
    }
    
    return metrics, poa

def analyze_energy_coefficients(timestamp, log_folder="test_logs"):
    """Analyze PoA for different energy coefficients"""
    energy_coeffs = [0.5, 1, 2]
    results = []
    
    for coeff in energy_coeffs:
        # Construct filenames
        ra2c_file = f"{log_folder}/test_log_RA2C_fullmesh_n{coeff}_c2_{timestamp}.csv"
        ra2cfull_file = f"{log_folder}/test_log_RA2CFull_fullmesh_n{coeff}_c2_{timestamp}.csv"
        
        if os.path.exists(ra2c_file) and os.path.exists(ra2cfull_file):
            metrics, poa = calculate_poa(ra2c_file, ra2cfull_file)
            
            results.append({
                'energy_coeff': coeff,
                **metrics,
                **poa
            })
        else:
            print(f"Warning: Missing files for energy coefficient {coeff}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print formatted results
    print("\nPrice of Anarchy Analysis for Different Energy Coefficients")
    print("=" * 80)
    for _, row in results_df.iterrows():
        print(f"\nEnergy Coefficient: {row['energy_coeff']}")
        print("-" * 40)
        print(f"Reward PoA: {row['reward_poa']:.4f}")
        print(f"Energy PoA: {row['energy_poa']:.4f}")
        print(f"AoI PoA: {row['aoi_poa']:.4f}")
        print(f"\nDetailed Metrics:")
        print(f"RA2C   - Reward: {row['ra2c_reward']:.4f}, Energy: {row['ra2c_energy']:.4f}, AoI: {row['ra2c_aoi']:.4f}")
        print(f"RA2CFull - Reward: {row['ra2cfull_reward']:.4f}, Energy: {row['ra2cfull_energy']:.4f}, AoI: {row['ra2cfull_aoi']:.4f}")
    
    return results_df

# Usage example:
timestamp = "20250205_043137"  # Replace with your actual timestamp
results_df = analyze_energy_coefficients(timestamp)

# Optional: Save results to CSV
results_df.to_csv(f"test_logs/poa_analysis_{timestamp}.csv", index=False)
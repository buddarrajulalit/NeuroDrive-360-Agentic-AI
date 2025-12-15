"""Generate synthetic automotive telematics dataset.

This script generates a realistic synthetic dataset with 50,000 rows containing
automotive sensor data including speed, engine temperature, vibration, battery
voltage, mileage, and a binary fault label.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_synthetic_telematics_dataset(
    n_rows: int = 50000, 
    random_seed: int = 42,
    output_path: str = "telematics_data.csv"
) -> pd.DataFrame:
    """Generate synthetic automotive telematics dataset.

    The fault label is dependent on abnormal values in:
    - Engine temperature (critical threshold > 105°C or < 80°C)
    - Vibration (critical threshold > 8.0 g-force)
    - Battery voltage (critical threshold < 11.5V or > 14.5V)

    Args:
        n_rows: Number of rows to generate (default: 50000).
        random_seed: Random seed for reproducibility (default: 42).
        output_path: Path to save the generated CSV file (default: "telematics_data.csv").

    Returns:
        pandas.DataFrame: Generated dataset with columns:
            - timestamp: Timestamp of the reading
            - vehicle_id: Unique vehicle identifier
            - speed: Vehicle speed in km/h (0-150)
            - engine_temperature: Engine temperature in Celsius (70-120, with outliers)
            - vibration: Vibration level in g-force (0-10, with outliers)
            - battery_voltage: Battery voltage in volts (11-15, with outliers)
            - mileage: Cumulative vehicle mileage in km (0-200000)
            - fault_label: Binary label (0=normal, 1=fault)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Generate base timestamps (last 30 days)
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [
        start_date + timedelta(
            seconds=random.randint(0, 30 * 24 * 60 * 60)
        ) for _ in range(n_rows)
    ]
    timestamps.sort()
    
    # Generate vehicle IDs (simulate 100 vehicles)
    vehicle_ids = [f"VEH-{str(i).zfill(3)}" for i in range(1, 101)]
    vehicle_id_list = np.random.choice(vehicle_ids, size=n_rows)
    
    # Generate speed (km/h) - normal distribution, 0-150 km/h
    # Most vehicles drive at moderate speeds (30-80 km/h)
    speed = np.clip(
        np.random.normal(loc=55, scale=25, size=n_rows),
        0, 150
    ).round(2)
    
    # Generate mileage (cumulative, increasing over time)
    # Start each vehicle at random mileage between 0-150000 km
    base_mileage = {}
    mileage_list = []
    for vid in vehicle_id_list:
        if vid not in base_mileage:
            base_mileage[vid] = random.uniform(0, 150000)
        # Increment mileage slightly (simulate driving)
        increment = random.uniform(0.1, 50)  # km per reading
        base_mileage[vid] += increment
        mileage_list.append(min(base_mileage[vid], 200000))
    
    mileage = np.array(mileage_list).round(2)
    
    # Generate engine temperature (°C)
    # Normal operating range: 85-95°C, but include outliers
    engine_temp_normal = np.random.normal(loc=90, scale=5, size=int(n_rows * 0.85))
    engine_temp_abnormal = np.concatenate([
        np.random.normal(loc=105, scale=3, size=int(n_rows * 0.08)),  # Overheating
        np.random.normal(loc=75, scale=3, size=int(n_rows * 0.07))    # Underheating
    ])
    engine_temperature = np.clip(
        np.concatenate([engine_temp_normal, engine_temp_abnormal]),
        70, 120
    )[:n_rows]
    np.random.shuffle(engine_temperature)
    engine_temperature = engine_temperature.round(2)
    
    # Generate vibration (g-force)
    # Normal range: 1-5 g, but include outliers
    vibration_normal = np.random.gamma(shape=2, scale=1.5, size=int(n_rows * 0.85))
    vibration_abnormal = np.random.gamma(shape=8, scale=1.5, size=int(n_rows * 0.15))
    vibration = np.clip(
        np.concatenate([vibration_normal, vibration_abnormal]),
        0, 10
    )[:n_rows]
    np.random.shuffle(vibration)
    vibration = vibration.round(3)
    
    # Generate battery voltage (volts)
    # Normal range: 12-14V, but include outliers
    battery_normal = np.random.normal(loc=12.8, scale=0.5, size=int(n_rows * 0.85))
    battery_low = np.random.normal(loc=11.2, scale=0.3, size=int(n_rows * 0.08))  # Low voltage
    battery_high = np.random.normal(loc=14.8, scale=0.3, size=int(n_rows * 0.07))  # Overcharging
    battery_voltage = np.clip(
        np.concatenate([battery_normal, battery_low, battery_high]),
        11, 15
    )[:n_rows]
    np.random.shuffle(battery_voltage)
    battery_voltage = battery_voltage.round(2)
    
    # Generate fault labels based on abnormal conditions
    # Fault conditions:
    # 1. Engine temperature > 105°C OR < 80°C (critical thresholds)
    # 2. Vibration > 8.0 g-force (critical threshold)
    # 3. Battery voltage < 11.5V OR > 14.5V (critical thresholds)
    fault_label = np.where(
        (engine_temperature > 105) | (engine_temperature < 80) |
        (vibration > 8.0) |
        (battery_voltage < 11.5) | (battery_voltage > 14.5),
        1, 0
    )
    
    # Add some noise - 2% false positives (normal conditions flagged as fault)
    false_positive_indices = np.random.choice(
        np.where(fault_label == 0)[0],
        size=int(n_rows * 0.02),
        replace=False
    )
    fault_label[false_positive_indices] = 1
    
    # Add some noise - 1% false negatives (fault conditions not flagged)
    false_negative_indices = np.random.choice(
        np.where(fault_label == 1)[0],
        size=int(n_rows * 0.01),
        replace=False
    )
    fault_label[false_negative_indices] = 0
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'vehicle_id': vehicle_id_list,
        'speed': speed,
        'engine_temperature': engine_temperature,
        'vibration': vibration,
        'battery_voltage': battery_voltage,
        'mileage': mileage,
        'fault_label': fault_label.astype(int)
    })
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully!")
    print(f"  - Total rows: {len(df):,}")
    print(f"  - Vehicles: {df['vehicle_id'].nunique()}")
    print(f"  - Fault instances: {df['fault_label'].sum():,} ({df['fault_label'].mean()*100:.2f}%)")
    print(f"  - Normal instances: {(df['fault_label']==0).sum():,} ({(df['fault_label']==0).mean()*100:.2f}%)")
    print(f"  - Saved to: {output_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    return df


if __name__ == "__main__":
    # Generate the dataset
    dataset = generate_synthetic_telematics_dataset(
        n_rows=50000,
        random_seed=42,
        output_path="telematics_data.csv"
    )
    
    print("\nFirst 10 rows:")
    print(dataset.head(10))


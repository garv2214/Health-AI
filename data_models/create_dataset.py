"""
Medical Emergency Dataset Generator
Creates realistic healthcare data for training emergency prediction models
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

def generate_medical_dataset(n_samples=10000):
    """Generate realistic medical emergency dataset."""
    
    print(f"📊 Generating {n_samples} medical records...")
    
    conditions = [
        'hypertension', 'diabetes', 'heart_disease', 'stroke', 
        'cancer', 'kidney_disease', 'liver_disease', 'asthma',
        'arthritis', 'obesity'
    ]
    
    data = []
    
    for i in range(n_samples):
        # Demographics
        age = max(18, min(100, np.random.normal(50, 20)))
        weight = max(30, min(150, np.random.normal(70, 15)))
        
        # Vital signs
        heart_rate = max(40, min(220, np.random.normal(80, 15)))
        
        base_bp = 120 + (age - 30) * 0.5
        systolic = max(80, min(250, np.random.normal(base_bp, 15)))
        diastolic = max(50, min(170, np.random.normal(systolic - 60, 10)))
        
        oxygen_saturation = max(75, min(100, np.random.normal(97, 2)))
        temperature = max(35, min(42, np.random.normal(37.0, 0.5)))
        respiration_rate = max(10, min(40, np.random.normal(16, 3)))
        
        # Medical history
        num_conditions = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
        history = random.sample(conditions, num_conditions) if num_conditions > 0 else []
        
        # Emergency indicators
        # Ensure probabilities sum to 1 (avoid runtime error if edited values drift)
        pain_probs = np.array([0.15, 0.15, 0.12, 0.10, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.07], dtype=float)
        pain_probs = pain_probs / pain_probs.sum()
        pain_level = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            p=pain_probs
        )
        consciousness_score = max(0, min(10, np.random.normal(9, 1.5)))
        
        # Emergency risk calculation
        emergency_risk = 0
        if heart_rate > 120 or heart_rate < 50:
            emergency_risk += 0.2
        if systolic > 180 or diastolic > 110:
            emergency_risk += 0.25
        if oxygen_saturation < 92:
            emergency_risk += 0.3
        if temperature > 38.5 or temperature < 36:
            emergency_risk += 0.15
        if pain_level >= 8:
            emergency_risk += 0.15
        if consciousness_score < 7:
            emergency_risk += 0.25
        if respiration_rate > 25 or respiration_rate < 10:
            emergency_risk += 0.2
        if 'heart_disease' in history:
            emergency_risk += 0.1
        if 'stroke' in history:
            emergency_risk += 0.1
        if age > 65:
            emergency_risk += 0.1
        
        emergency_risk += np.random.normal(0, 0.05)
        is_emergency = 1 if emergency_risk > 0.5 else 0
        
        data.append({
            'patient_id': f'P{i+1}',
            'age': round(age, 1),
            'weight_kg': round(weight, 1),
            'heart_rate': round(heart_rate, 1),
            'blood_pressure_systolic': round(systolic, 1),
            'blood_pressure_diastolic': round(diastolic, 1),
            'blood_pressure': f"{round(systolic)}/{round(diastolic)}",
            'oxygen_saturation': round(oxygen_saturation, 1),
            'temperature': round(temperature, 1),
            'respiration_rate': round(respiration_rate, 1),
            'pain_level': pain_level,
            'consciousness_score': round(consciousness_score, 1),
            'num_conditions': num_conditions,
            'history': history,
            'is_emergency': is_emergency
        })
    
    df = pd.DataFrame(data)
    
    print(f"✅ Dataset generated!")
    print(f"   Total: {len(df)} samples")
    print(f"   Emergency: {df['is_emergency'].sum()} ({df['is_emergency'].mean()*100:.1f}%)")
    print(f"   Stable: {len(df) - df['is_emergency'].sum()} ({(1-df['is_emergency'].mean())*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    import os

    df = generate_medical_dataset(10000)

    out_dir = 'data'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'medical_emergency_dataset.csv')

    df.to_csv(out_path, index=False)
    print(f"\n💾 Saved to: {out_path}")


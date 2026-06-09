import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

RANDOM_SEED = 42
TARGET_COL = "target"
ID_COL = "id"

print(f"Constants initialized:")
print(f"  RANDOM_SEED: {RANDOM_SEED}")
print(f"  TARGET_COL: {TARGET_COL}")
print(f"  ID_COL: {ID_COL}")


import pandas as pd

RANDOM_SEED = 42
TARGET_COL = "target"
ID_COL = "id"

# Load data
train_raw = pd.read_csv("training_data.csv")
test_raw = pd.read_csv("test_data.csv")

# Separate features and target
X_full = train_raw.drop(columns=[TARGET_COL])
y_full = train_raw[TARGET_COL]

# Store test IDs before dropping
test_ids = test_raw[ID_COL]

print(f"Data loaded successfully:")
print(f"  Train shape: {train_raw.shape}")
print(f"  Test shape: {test_raw.shape}")
print(f"  Features: {X_full.shape[1]}")
print(f"  Target distribution: {y_full.value_counts().to_dict()}")


import pandas as pd

ID_COL = "id"

# Detect categorical columns
cat_cols_detected = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols_filtered = [c for c in cat_cols_detected if c != ID_COL]

# Convert categorical columns to string
for col in cat_cols_filtered:
    X_full[col] = X_full[col].astype("string")
    test_raw[col] = test_raw[col].astype("string")

# Drop ID from features
X_processed = X_full.drop(columns=[ID_COL])
test_processed = test_raw.drop(columns=[ID_COL])

print(f"Categorical processing complete:")
print(f"  Categorical columns detected: {len(cat_cols_filtered)}")
print(f"  Columns: {cat_cols_filtered}")
print(f"  X shape after dropping ID: {X_processed.shape}")
print(f"  Test shape after dropping ID: {test_processed.shape}")

from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# First split: 80% (temp) and 20% (holdout)
X_temp, X_holdout, y_temp, y_holdout = train_test_split(
    X_processed, y_full,
    test_size=0.20,
    stratify=y_full,
    random_state=RANDOM_SEED
)

# Second split: 75% of temp (train) and 25% of temp (valid)
# This gives us 60% train, 20% valid, 20% holdout overall
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    stratify=y_temp,
    random_state=RANDOM_SEED
)

print(f"Train-validation split (60/20/20) complete:")
print(f"  Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X_processed)*100:.1f}%)")
print(f"  Valid: {X_valid.shape[0]:,} samples ({X_valid.shape[0]/len(X_processed)*100:.1f}%)")
print(f"  Holdout: {X_holdout.shape[0]:,} samples ({X_holdout.shape[0]/len(X_processed)*100:.1f}%)")
print(f"  Target distribution - Train: {y_train.value_counts().to_dict()}")
print(f"  Target distribution - Valid: {y_valid.value_counts().to_dict()}")


from catboost import Pool

# Get categorical feature indices (store as exportable variable)
cat_indices_list = [X_processed.columns.get_loc(col) for col in cat_cols_filtered]

print(f"Categorical feature processing:")
print(f"  Categorical feature indices: {cat_indices_list}")
print(f"  Number of categorical features: {len(cat_indices_list)}")


from catboost import CatBoostClassifier, Pool

RANDOM_SEED = 42

# Create pools locally (not exported)
_train_pool = Pool(X_train, y_train, cat_features=cat_indices_list)
_valid_pool = Pool(X_valid, y_valid, cat_features=cat_indices_list)

# Train initial model with early stopping
initial_model = CatBoostClassifier(
    iterations=3000,
    depth=8,
    learning_rate=0.02,
    loss_function="Logloss",
    eval_metric="AUC",
    l2_leaf_reg=10,
    random_strength=2.0,
    bootstrap_type="Bernoulli",
    subsample=0.85,
    rsm=0.85,
    one_hot_max_size=2,
    max_ctr_complexity=3,
    grow_policy="Lossguide",
    early_stopping_rounds=150,
    random_seed=RANDOM_SEED,
    thread_count=-1,
    verbose=100
)

initial_model.fit(_train_pool, eval_set=_valid_pool)

best_iteration = initial_model.best_iteration_ + 1

print(f"\nInitial model training complete:")
print(f"  Best iteration: {best_iteration}")
print(f"  Best validation AUC: {initial_model.best_score_['validation']['AUC']:.6f}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset overview
print("=" * 70)
print("📊 DATASET OVERVIEW")
print("=" * 70)
print(f"\n🔹 Training Data:")
print(f"   • Rows: {len(train_raw):,}")
print(f"   • Columns: {len(train_raw.columns)}")
print(f"   • Features: {len(train_raw.columns) - 2} (50 features + id + target)")
print(f"\n🔹 Test Data:")
print(f"   • Rows: {len(test_raw):,}")
print(f"   • Columns: {len(test_raw.columns)}")
print(f"   • Features: {len(test_raw.columns) - 1} (50 features + id)")

# Target distribution
target_counts = y_full.value_counts().sort_index()
target_pct = (target_counts / len(y_full) * 100)

print(f"\n🎯 Target Variable Distribution:")
print(f"   • Class 0: {target_counts[0]:,} samples ({target_pct[0]:.2f}%)")
print(f"   • Class 1: {target_counts[1]:,} samples ({target_pct[1]:.2f}%)")
print(f"   • Imbalance Ratio: {target_counts[0]/target_counts[1]:.2f}:1")

# Feature types analysis
feature_cols = [col for col in train_raw.columns if col not in ['id', 'target']]
int_features = train_raw[feature_cols].select_dtypes(include=['int64']).columns
float_features = train_raw[feature_cols].select_dtypes(include=['float64']).columns

print(f"\n📋 Feature Types:")
print(f"   • Integer features: {len(int_features)}")
print(f"   • Float features: {len(float_features)}")

# Missing values analysis
missing_train = train_raw[feature_cols].isnull().sum()
missing_test = test_raw[feature_cols].isnull().sum()
features_with_missing_train = (missing_train > 0).sum()
features_with_missing_test = (missing_test > 0).sum()

print(f"\n🔍 Missing Values:")
print(f"   • Training set: {features_with_missing_train} features have missing values")
print(f"   • Test set: {features_with_missing_test} features have missing values")
print(f"   • Total missing (train): {missing_train.sum():,} ({missing_train.sum() / (len(train_raw) * len(feature_cols)) * 100:.2f}%)")
print(f"   • Total missing (test): {missing_test.sum():,} ({missing_test.sum() / (len(test_raw) * len(feature_cols)) * 100:.2f}%)")

if features_with_missing_train > 0:
    top_missing = missing_train[missing_train > 0].sort_values(ascending=False).head(5)
    print(f"\n   Top features with missing values (train):")
    for feat, count in top_missing.items():
        pct = (count / len(train_raw)) * 100
        print(f"      • {feat}: {count:,} ({pct:.2f}%)")

# Basic statistics
print(f"\n📈 Feature Value Ranges (sample):")
sample_features = feature_cols[:5]
for feat in sample_features:
    min_val = train_raw[feat].min()
    max_val = train_raw[feat].max()
    mean_val = train_raw[feat].mean()
    print(f"   • {feat}: [{min_val:.2f}, {max_val:.2f}] (mean: {mean_val:.2f})")

print("\n" + "=" * 70)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Zerve design system colors
ZERVE_BG = '#1D1D20'
ZERVE_TEXT = '#fbfbff'
ZERVE_SECONDARY = '#909094'
ZERVE_COLORS = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#1F77B4', '#9467BD', '#8C564B']
ZERVE_HIGHLIGHT = '#ffd400'
ZERVE_SUCCESS = '#17b26a'
ZERVE_WARNING = '#f04438'

# Set style for all plots
plt.rcParams['figure.facecolor'] = ZERVE_BG
plt.rcParams['axes.facecolor'] = ZERVE_BG
plt.rcParams['axes.edgecolor'] = ZERVE_SECONDARY
plt.rcParams['axes.labelcolor'] = ZERVE_TEXT
plt.rcParams['text.color'] = ZERVE_TEXT
plt.rcParams['xtick.color'] = ZERVE_TEXT
plt.rcParams['ytick.color'] = ZERVE_TEXT
plt.rcParams['legend.facecolor'] = ZERVE_BG
plt.rcParams['legend.edgecolor'] = ZERVE_SECONDARY

# 1. Target Distribution with Zerve styling
target_dist_fig = plt.figure(figsize=(10, 6))
ax = plt.gca()
_counts = y_full.value_counts().sort_index()
_labels = ['Class 0\n(No Event)', 'Class 1\n(Event)']
bars = ax.bar(_labels, _counts, color=[ZERVE_COLORS[0], ZERVE_WARNING], edgecolor=ZERVE_TEXT, linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}\n({height/len(y_full)*100:.1f}%)',
            ha='center', va='bottom', color=ZERVE_TEXT, fontsize=11, fontweight='bold')

ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_title('Target Variable Distribution\nHighly Imbalanced Dataset (26.4:1 ratio)', 
             fontsize=14, fontweight='bold', pad=20, color=ZERVE_TEXT)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
print("✓ Target distribution chart created")

# 2. Missing Values Heatmap
missing_viz_fig = plt.figure(figsize=(12, 6))
_features_with_missing = missing_train[missing_train > 0].sort_values(ascending=False)
ax = plt.gca()
bars = ax.barh(range(len(_features_with_missing)), 
               (_features_with_missing / len(train_raw) * 100).values,
               color=ZERVE_COLORS[3], edgecolor=ZERVE_TEXT, linewidth=1)
ax.set_yticks(range(len(_features_with_missing)))
ax.set_yticklabels(_features_with_missing.index, fontsize=10, color=ZERVE_TEXT)
ax.set_xlabel('Missing Percentage (%)', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_title('Features with Missing Values\nTop features require careful handling', 
             fontsize=14, fontweight='bold', pad=20, color=ZERVE_TEXT)

# Add percentage labels
for i, (feat, val) in enumerate(_features_with_missing.items()):
    pct = val / len(train_raw) * 100
    ax.text(pct + 1, i, f'{pct:.1f}%', va='center', color=ZERVE_TEXT, fontsize=9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.invert_yaxis()
plt.tight_layout()
print("✓ Missing values chart created")

# 3. Feature Distribution Samples
feature_dist_fig = plt.figure(figsize=(14, 10))
_sample_cols = [col for col in feature_cols if col not in ['feature_39', 'feature_8', 'feature_45']][:8]
for idx, col in enumerate(_sample_cols, 1):
    ax = plt.subplot(2, 4, idx)
    _data = train_raw[col].dropna()
    
    # Use histogram with KDE overlay
    ax.hist(_data, bins=50, color=ZERVE_COLORS[idx % len(ZERVE_COLORS)], 
            alpha=0.7, edgecolor=ZERVE_TEXT, linewidth=0.5)
    
    ax.set_title(col, fontsize=11, fontweight='bold', color=ZERVE_TEXT)
    ax.set_xlabel('Value', fontsize=9, color=ZERVE_TEXT)
    ax.set_ylabel('Frequency', fontsize=9, color=ZERVE_TEXT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

plt.suptitle('Feature Value Distributions\nSample of 8 features showing varied patterns', 
             fontsize=14, fontweight='bold', y=0.995, color=ZERVE_TEXT)
plt.tight_layout()
print("✓ Feature distributions chart created")

# 4. Data Quality Overview
quality_fig = plt.figure(figsize=(10, 6))
_quality_metrics = {
    'Complete Features': len(feature_cols) - features_with_missing_train,
    'Features with\nMissing Data': features_with_missing_train,
    'Integer Features': len(int_features),
    'Float Features': len(float_features)
}
ax = plt.gca()
bars = ax.bar(range(len(_quality_metrics)), list(_quality_metrics.values()),
              color=[ZERVE_SUCCESS, ZERVE_WARNING, ZERVE_COLORS[0], ZERVE_COLORS[4]],
              edgecolor=ZERVE_TEXT, linewidth=1.5)
ax.set_xticks(range(len(_quality_metrics)))
ax.set_xticklabels(list(_quality_metrics.keys()), fontsize=11, color=ZERVE_TEXT)
ax.set_ylabel('Count', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_title('Dataset Quality Overview\n50 features across integer and float types', 
             fontsize=14, fontweight='bold', pad=20, color=ZERVE_TEXT)

# Add value labels
for i, (key, val) in enumerate(_quality_metrics.items()):
    ax.text(i, val, f'{val}', ha='center', va='bottom', 
            color=ZERVE_TEXT, fontsize=11, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
print("✓ Data quality overview chart created")

print("\n📊 Created 4 data distribution visualizations")
print("   • Target distribution showing class imbalance")
print("   • Missing values analysis across features") 
print("   • Sample feature distributions")
print("   • Data quality metrics")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Zerve design system colors
ZERVE_BG = '#1D1D20'
ZERVE_TEXT = '#fbfbff'
ZERVE_SECONDARY = '#909094'
ZERVE_COLORS = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#1F77B4', '#9467BD', '#8C564B']
ZERVE_HIGHLIGHT = '#ffd400'
ZERVE_SUCCESS = '#17b26a'
ZERVE_WARNING = '#f04438'

# Set style
plt.rcParams['figure.facecolor'] = ZERVE_BG
plt.rcParams['axes.facecolor'] = ZERVE_BG
plt.rcParams['axes.edgecolor'] = ZERVE_SECONDARY
plt.rcParams['axes.labelcolor'] = ZERVE_TEXT
plt.rcParams['text.color'] = ZERVE_TEXT
plt.rcParams['xtick.color'] = ZERVE_TEXT
plt.rcParams['ytick.color'] = ZERVE_TEXT
plt.rcParams['legend.facecolor'] = ZERVE_BG
plt.rcParams['legend.edgecolor'] = ZERVE_SECONDARY

# 1. Feature Correlations with Target
correlation_fig = plt.figure(figsize=(14, 8))

# Calculate correlations with target for numeric features
_feature_list = [col for col in train_raw.columns if col not in ['id', 'target']]
_correlations = []
for feat in _feature_list:
    _clean_data = train_raw[[feat, 'target']].dropna()
    if len(_clean_data) > 0:
        _corr, _ = pearsonr(_clean_data[feat], _clean_data['target'])
        _correlations.append((_corr, feat))

_correlations.sort(key=lambda x: abs(x[0]), reverse=True)
_top_corr = _correlations[:20]
_corr_values = [c[0] for c in _top_corr]
_corr_names = [c[1] for c in _top_corr]

ax = plt.gca()
_colors = [ZERVE_SUCCESS if c > 0 else ZERVE_WARNING for c in _corr_values]
bars = ax.barh(range(len(_corr_values)), _corr_values, color=_colors, edgecolor=ZERVE_TEXT, linewidth=1)
ax.set_yticks(range(len(_corr_names)))
ax.set_yticklabels(_corr_names, fontsize=9, color=ZERVE_TEXT)
ax.set_xlabel('Correlation with Target', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_title('Top 20 Features by Correlation with Target\nPositive (green) vs Negative (red) correlation', 
             fontsize=14, fontweight='bold', pad=20, color=ZERVE_TEXT)
ax.axvline(x=0, color=ZERVE_SECONDARY, linestyle='--', linewidth=1)

# Add value labels
for i, val in enumerate(_corr_values):
    ax.text(val + (0.002 if val > 0 else -0.002), i, f'{val:.3f}', 
            va='center', ha='left' if val > 0 else 'right', color=ZERVE_TEXT, fontsize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.invert_yaxis()
plt.tight_layout()
print("✓ Feature-target correlations chart created")

# 2. Target Distribution by Sample Features
target_by_feature_fig = plt.figure(figsize=(14, 10))
_key_features = [feat[1] for feat in _top_corr[:6]]

for idx, feat in enumerate(_key_features, 1):
    ax = plt.subplot(2, 3, idx)
    
    _data_class0 = train_raw[train_raw['target'] == 0][feat].dropna()
    _data_class1 = train_raw[train_raw['target'] == 1][feat].dropna()
    
    # Overlapping histograms
    ax.hist(_data_class0, bins=40, alpha=0.6, color=ZERVE_COLORS[0], 
            label='Class 0', edgecolor=ZERVE_TEXT, linewidth=0.3)
    ax.hist(_data_class1, bins=40, alpha=0.6, color=ZERVE_WARNING, 
            label='Class 1', edgecolor=ZERVE_TEXT, linewidth=0.3)
    
    ax.set_title(feat, fontsize=11, fontweight='bold', color=ZERVE_TEXT)
    ax.set_xlabel('Value', fontsize=9, color=ZERVE_TEXT)
    ax.set_ylabel('Frequency', fontsize=9, color=ZERVE_TEXT)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8, 
              facecolor=ZERVE_BG, edgecolor=ZERVE_SECONDARY, labelcolor=ZERVE_TEXT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

plt.suptitle('Target Distribution Across Top Correlated Features\nComparing Class 0 vs Class 1 distributions', 
             fontsize=14, fontweight='bold', y=0.995, color=ZERVE_TEXT)
plt.tight_layout()
print("✓ Target distribution by features chart created")

# 3. Feature Correlation Heatmap (Top Features)
heatmap_fig = plt.figure(figsize=(12, 10))
_top_features = [feat[1] for feat in _top_corr[:15]]
_corr_matrix = train_raw[_top_features].corr()

ax = plt.gca()
im = ax.imshow(_corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation', rotation=270, labelpad=20, color=ZERVE_TEXT, fontsize=11, fontweight='bold')
cbar.ax.yaxis.set_tick_params(color=ZERVE_TEXT)
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=ZERVE_TEXT)

ax.set_xticks(range(len(_top_features)))
ax.set_yticks(range(len(_top_features)))
ax.set_xticklabels(_top_features, rotation=45, ha='right', fontsize=9, color=ZERVE_TEXT)
ax.set_yticklabels(_top_features, fontsize=9, color=ZERVE_TEXT)
ax.set_title('Feature Correlation Heatmap\nTop 15 features most correlated with target', 
             fontsize=14, fontweight='bold', pad=20, color=ZERVE_TEXT)

plt.tight_layout()
print("✓ Feature correlation heatmap created")

print("\n📊 Created 3 correlation and target analysis visualizations")
print("   • Top 20 feature-target correlations")
print("   • Target distributions across key features")
print("   • Correlation heatmap of top features")



import pandas as pd
import numpy as np

# Load and analyze CatBoost training results
print("=" * 70)
print("🤖 MODEL PERFORMANCE ANALYSIS")
print("=" * 70)

# Read the submission file to analyze predictions
submission = pd.read_csv("ans.csv")

print(f"\n📊 Submission File Analysis:")
print(f"   • Total predictions: {len(submission):,}")
print(f"   • Probability range: [{submission['probability'].min():.6f}, {submission['probability'].max():.6f}]")
print(f"   • Mean probability: {submission['probability'].mean():.6f}")
print(f"   • Median probability: {submission['probability'].median():.6f}")
print(f"   • Std deviation: {submission['probability'].std():.6f}")

# Analyze probability distribution
print(f"\n📈 Probability Distribution:")
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(submission['probability'], p)
    print(f"   • {p}th percentile: {val:.6f}")

# Load error metrics
learn_errors = pd.read_csv('catboost_info/learn_error.tsv', sep='\t')
test_errors = pd.read_csv('catboost_info/test_error.tsv', sep='\t')

print(f"\n🔧 Training Data Structure:")
print(f"   • Training metrics columns: {list(learn_errors.columns)}")
print(f"   • Validation metrics columns: {list(test_errors.columns)}")

final_iteration = len(test_errors) - 1
final_learn_logloss = learn_errors['Logloss'].iloc[-1]
final_test_logloss = test_errors['Logloss'].iloc[-1]

# Check if AUC is available in test errors
if 'AUC' in test_errors.columns:
    final_test_auc = test_errors['AUC'].iloc[-1]
    best_iteration_idx = test_errors['AUC'].idxmax()
    best_test_auc = test_errors['AUC'].iloc[best_iteration_idx]
    
    print(f"\n🎯 Training Performance:")
    print(f"   • Total iterations: {final_iteration + 1}")
    print(f"   • Final validation AUC: {final_test_auc:.6f}")
    print(f"   • Final train Logloss: {final_learn_logloss:.6f}")
    print(f"   • Final validation Logloss: {final_test_logloss:.6f}")
    print(f"   • Gini coefficient: {2 * final_test_auc - 1:.6f}")
    
    print(f"\n⭐ Best Performance (based on validation AUC):") 
    print(f"   • Best iteration: {best_iteration_idx}")
    print(f"   • Best validation AUC: {best_test_auc:.6f}")
    print(f"   • Gini coefficient: {2 * best_test_auc - 1:.6f}")
    
    # Analyze training curves
    print(f"\n📉 Training Curve Analysis:")
    print(f"   • First 10 iters - Val AUC: {test_errors['AUC'].iloc[9]:.6f}")
    print(f"   • First 50 iters - Val AUC: {test_errors['AUC'].iloc[49]:.6f}")
    if len(test_errors) > 100:
        print(f"   • First 100 iters - Val AUC: {test_errors['AUC'].iloc[99]:.6f}")
    
    model_metrics = {
        'final_test_auc': final_test_auc,
        'best_test_auc': best_test_auc,
        'best_iteration': best_iteration_idx,
        'gini': 2 * final_test_auc - 1
    }
else:
    print(f"\n🎯 Training Performance:")
    print(f"   • Total iterations: {final_iteration + 1}")
    print(f"   • Final train Logloss: {final_learn_logloss:.6f}")
    print(f"   • Final validation Logloss: {final_test_logloss:.6f}")
    print(f"   • Lower logloss is better")
    
    # Find best iteration by lowest validation logloss
    best_iteration_idx = test_errors['Logloss'].idxmin()
    best_test_logloss = test_errors['Logloss'].iloc[best_iteration_idx]
    
    print(f"\n⭐ Best Performance (based on validation Logloss):")
    print(f"   • Best iteration: {best_iteration_idx}")
    print(f"   • Best validation Logloss: {best_test_logloss:.6f}")
    
    model_metrics = {
        'final_test_logloss': final_test_logloss,
        'best_test_logloss': best_test_logloss,
        'best_iteration': best_iteration_idx
    }

print("\n" + "=" * 70)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Zerve design system colors
ZERVE_BG = '#1D1D20'
ZERVE_TEXT = '#fbfbff'
ZERVE_SECONDARY = '#909094'
ZERVE_COLORS = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#1F77B4', '#9467BD', '#8C564B']
ZERVE_HIGHLIGHT = '#ffd400'
ZERVE_SUCCESS = '#17b26a'
ZERVE_WARNING = '#f04438'

# Set style
plt.rcParams['figure.facecolor'] = ZERVE_BG
plt.rcParams['axes.facecolor'] = ZERVE_BG
plt.rcParams['axes.edgecolor'] = ZERVE_SECONDARY
plt.rcParams['axes.labelcolor'] = ZERVE_TEXT
plt.rcParams['text.color'] = ZERVE_TEXT
plt.rcParams['xtick.color'] = ZERVE_TEXT
plt.rcParams['ytick.color'] = ZERVE_TEXT
plt.rcParams['legend.facecolor'] = ZERVE_BG
plt.rcParams['legend.edgecolor'] = ZERVE_SECONDARY

# 1. Training Learning Curves (AUC)
learning_curve_fig = plt.figure(figsize=(14, 6))
ax = plt.gca()

ax.plot(test_errors['iter'], test_errors['AUC'], 
        color=ZERVE_COLORS[0], linewidth=2, label='Validation AUC')
ax.axhline(y=best_test_auc, color=ZERVE_SUCCESS, linestyle='--', linewidth=1.5, 
           label=f'Best AUC: {best_test_auc:.4f} (iter {best_iteration_idx})')
ax.axhline(y=0.5, color=ZERVE_SECONDARY, linestyle=':', linewidth=1, 
           label='Random Classifier (0.5)')

ax.set_xlabel('Iteration', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_title('Model Training Progress: AUC Score\nValidation performance across iterations', 
             fontsize=14, fontweight='bold', pad=20, color=ZERVE_TEXT)
ax.legend(loc='lower right', framealpha=0.9, fontsize=10, 
          facecolor=ZERVE_BG, edgecolor=ZERVE_SECONDARY, labelcolor=ZERVE_TEXT)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2, color=ZERVE_SECONDARY)
plt.tight_layout()
print("✓ AUC learning curve created")

# 2. Logloss Learning Curves
logloss_curve_fig = plt.figure(figsize=(14, 6))
ax = plt.gca()

ax.plot(learn_errors['iter'], learn_errors['Logloss'], 
        color=ZERVE_COLORS[1], linewidth=2, label='Training Logloss', alpha=0.8)
ax.plot(test_errors['iter'], test_errors['Logloss'], 
        color=ZERVE_COLORS[0], linewidth=2, label='Validation Logloss')

ax.set_xlabel('Iteration', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_ylabel('Logloss', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_title('Model Training Progress: Logloss\nMonitoring overfitting via train vs validation loss', 
             fontsize=14, fontweight='bold', pad=20, color=ZERVE_TEXT)
ax.legend(loc='upper right', framealpha=0.9, fontsize=10, 
          facecolor=ZERVE_BG, edgecolor=ZERVE_SECONDARY, labelcolor=ZERVE_TEXT)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2, color=ZERVE_SECONDARY)
plt.tight_layout()
print("✓ Logloss learning curves created")

# 3. Model Performance Metrics Summary
metrics_fig = plt.figure(figsize=(10, 6))
ax = plt.gca()

_metrics = {
    'Best\nValidation AUC': best_test_auc,
    'Final\nValidation AUC': final_test_auc,
    'Gini\nCoefficient': 2 * best_test_auc - 1,
    'Final Train\nLogloss': final_learn_logloss,
    'Final Val\nLogloss': final_test_logloss
}

_colors_map = [ZERVE_SUCCESS, ZERVE_COLORS[0], ZERVE_HIGHLIGHT, ZERVE_COLORS[1], ZERVE_COLORS[3]]
bars = ax.bar(range(len(_metrics)), list(_metrics.values()), 
              color=_colors_map, edgecolor=ZERVE_TEXT, linewidth=1.5)

ax.set_xticks(range(len(_metrics)))
ax.set_xticklabels(list(_metrics.keys()), fontsize=10, color=ZERVE_TEXT)
ax.set_ylabel('Score', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_title('Model Performance Summary\nKey metrics from CatBoost training', 
             fontsize=14, fontweight='bold', pad=20, color=ZERVE_TEXT)

# Add value labels
for i, (key, val) in enumerate(_metrics.items()):
    ax.text(i, val, f'{val:.4f}', ha='center', va='bottom', 
            color=ZERVE_TEXT, fontsize=10, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
print("✓ Performance metrics summary created")

# 4. Prediction Distribution
pred_dist_fig = plt.figure(figsize=(12, 6))
ax = plt.gca()

ax.hist(submission['probability'], bins=100, color=ZERVE_COLORS[0], 
        alpha=0.7, edgecolor=ZERVE_TEXT, linewidth=0.5)
ax.axvline(x=submission['probability'].mean(), color=ZERVE_WARNING, 
           linestyle='--', linewidth=2, label=f'Mean: {submission["probability"].mean():.4f}')
ax.axvline(x=submission['probability'].median(), color=ZERVE_SUCCESS, 
           linestyle='--', linewidth=2, label=f'Median: {submission["probability"].median():.4f}')

ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_title('Test Set Prediction Distribution\nModel probability predictions for test data', 
             fontsize=14, fontweight='bold', pad=20, color=ZERVE_TEXT)
ax.legend(loc='upper right', framealpha=0.9, fontsize=10, 
          facecolor=ZERVE_BG, edgecolor=ZERVE_SECONDARY, labelcolor=ZERVE_TEXT)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
print("✓ Prediction distribution chart created")

# 5. Early Stopping Analysis
early_stop_fig = plt.figure(figsize=(12, 6))
ax = plt.gca()

_window = 50
_auc_smooth = test_errors['AUC'].rolling(window=_window, center=True).mean()
ax.plot(test_errors['iter'], test_errors['AUC'], 
        color=ZERVE_COLORS[0], linewidth=1, alpha=0.3, label='Raw AUC')
ax.plot(test_errors['iter'], _auc_smooth, 
        color=ZERVE_COLORS[0], linewidth=2, label=f'Smoothed AUC ({_window}-iter window)')
ax.axvline(x=best_iteration_idx, color=ZERVE_SUCCESS, linestyle='--', 
           linewidth=2, label=f'Best Iteration: {best_iteration_idx}')

ax.set_xlabel('Iteration', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold', color=ZERVE_TEXT)
ax.set_title('Early Stopping Analysis\nSmoothed performance showing optimal stopping point', 
             fontsize=14, fontweight='bold', pad=20, color=ZERVE_TEXT)
ax.legend(loc='lower right', framealpha=0.9, fontsize=10, 
          facecolor=ZERVE_BG, edgecolor=ZERVE_SECONDARY, labelcolor=ZERVE_TEXT)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2, color=ZERVE_SECONDARY)
plt.tight_layout()
print("✓ Early stopping analysis chart created")

print("\n📊 Created 5 model performance visualizations")
print("   • AUC learning curve showing validation progress")
print("   • Logloss curves comparing train vs validation")
print("   • Performance metrics summary bar chart")
print("   • Test prediction distribution histogram")
print("   • Early stopping analysis with smoothing")



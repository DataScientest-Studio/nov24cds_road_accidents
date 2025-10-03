#Load the packages 
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
raw_path = "../data/raw"
processed_path = "../data/processed"
output_file = os.path.join(processed_path, "accidents_merged_2005_2023.csv")

#Some files have different delimiter and a typical French encoder ISO-8859-1
def robust_read_csv(fpath, encoding="ISO-8859-1"): 
    for delim in [",", ";", "\t"]:
        df = pd.read_csv(fpath, delimiter=delim, encoding=encoding, on_bad_lines="skip", low_memory=False)
        df.columns = df.columns.str.strip().str.lower().str.replace('"', '')
        # Rename for accident_id
        if 'accident_id' in df.columns:
            df = df.rename(columns={'accident_id': 'num_acc'}) #some datasets had different name for the num_acc key column
        if 'num_acc' in df.columns:
            return df
    print(f"!! Could not properly split columns for {fpath}, got: {df.columns.tolist()}")
    return df

all_years = []
for year in range(2005, 2024):
    files = {
        "carac": os.path.join(raw_path, f"caracteristiques_{year}.csv"),
        "lieux": os.path.join(raw_path, f"lieux_{year}.csv"),
        "vehic": os.path.join(raw_path, f"vehicules_{year}.csv"),
        "usag": os.path.join(raw_path, f"usagers_{year}.csv"),
    }
    dfs = {k: robust_read_csv(fpath) for k, fpath in files.items()}

    if all('num_acc' in df.columns for df in dfs.values()):
        df = dfs["carac"].merge(dfs["lieux"], on="num_acc", how="inner", suffixes=('', '_lieux')) #Keep only rows that have a match in both DataFrames
        df = df.merge(dfs["vehic"], on="num_acc", how="inner", suffixes=('', '_vehic'))
        df = df.merge(dfs["usag"], on="num_acc", how="inner", suffixes=('', '_usag'))
        df["an"] = year  # Replace 'an' column with the current year for unification, for some years it was 5 instead of 2005
        all_years.append(df)
    else:
        print(f"Skipping year {year}: 'num_acc' not found in all files.")

# Combine all years and save
merged_df = pd.concat(all_years, ignore_index=True)
os.makedirs(processed_path, exist_ok=True)
merged_df.to_csv(output_file, index=False)
print(f"Merged file saved as: {output_file}")
print(merged_df.duplicated().sum())
merged_df = merged_df.drop_duplicates()
categorical_columns = [
    "lum", "agg", "int", "atm", "col", "catr", "circ", "prof", "plan", "surf",
    "infra", "situ", "env1", "senc", "catv", "catu", "grav", "sexe", "trajet",
    "secu", "locp", "etatp", "vosp", "obs", "obsm", "choc", "manv", "motor",
    "secu1", "secu2", "secu3", "place", "nbv", "actp", "pr", "pr1"
]

string_columns = [
    "com", "dep", "voie", "v2", "adr", "larrout",
    "lat", "long", "gps", "num_veh", "num_veh_usag", "id_vehicule",
    "id_vehicule_usag", "id_usager"
]

int_columns = ["an", "mois", "jour", "an_nais", "vma", "num_acc", "secu1", "secu2", "secu3", "secu"]
time_columns = ["hrmn"]

# Apply conversions
for col in categorical_columns:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].astype("category")

for col in string_columns:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].astype("string")

for col in int_columns:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').astype("Int64")

for col in time_columns:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].astype(str)

# Convert lartpc to float
if "lartpc" in merged_df.columns:
    merged_df["lartpc"] = pd.to_numeric(merged_df["lartpc"], errors="coerce")
import re

def clean_hrmn(val):
    if pd.isna(val):
        return "0000"
    
    val = str(val).strip()
    
    # Format: HH:MM
    if re.match(r"^\d{1,2}:\d{2}$", val):
        return val.replace(":", "").zfill(4)
    
    # Format: HHMM
    if val.isdigit() and len(val) in [3, 4]:
        return val.zfill(4)
    
    # Minute-only entry like "45" -> "0045"
    if val.isdigit() and len(val) <= 2:
        return "00" + val.zfill(2)
    
    # Invalid or unrecognized format
    return "0000"

merged_df["hrmn_clean"] = merged_df["hrmn"].apply(clean_hrmn)

merged_df["datetime"] = pd.to_datetime(
    merged_df["an"].astype(str) + "-" +
    merged_df["mois"].astype(str).str.zfill(2) + "-" +
    merged_df["jour"].astype(str).str.zfill(2) + " " +
    merged_df["hrmn_clean"].str[:2] + ":" + merged_df["hrmn_clean"].str[2:],
    format="%Y-%m-%d %H:%M",
    errors="coerce"
)

merged_df["hour"] = merged_df["datetime"].dt.hour.astype("Int64")
merged_df["dayofweek"] = merged_df["datetime"].dt.dayofweek.astype("Int64")
merged_df["weekday_name"] = merged_df["datetime"].dt.day_name().astype("category")
# Function to normalize value:
# - if string with comma, convert to float
# - if integer-looking, shift decimal
def normalize_gps_value(val):
    try:
        val_str = str(val).strip()
        if ',' in val_str:
            return float(val_str.replace(",", "."))
        val_float = float(val_str)
        if val_float > 1e5:
            shifted = str(int(val_float)).zfill(7)
            return float(shifted[:2] + '.' + shifted[2:])
        return val_float
    except:
        return np.nan

merged_df["lat_clean"] = merged_df["lat"].apply(normalize_gps_value)
merged_df["long_clean"] = merged_df["long"].apply(normalize_gps_value)

def combine_gps(row):
    lat = row["lat_clean"]
    lon = row["long_clean"]
    if pd.notna(lat) and pd.notna(lon) and not (lat == 0.0 and lon == 0.0):
        return f"{lat:.5f},{lon:.5f}"
    return np.nan

merged_df["gps_combined"] = merged_df.apply(combine_gps, axis=1)

merged_df["dep"] = pd.to_numeric(merged_df["dep"], errors="coerce")
merged_df["com"] = pd.to_numeric(merged_df["com"], errors="coerce")

def clean_dep(dep):
    try:
        dep_str = str(int(float(dep))).zfill(3)
        if len(dep_str) > 2 and dep_str.endswith("0"):
            return dep_str[:2]
        return dep_str
    except:
        return np.nan

def clean_com(com):
    try:
        return str(int(float(com)))[-3:].zfill(3)
    except:
        return np.nan

def build_insee(row):
    dep_clean = clean_dep(row["dep"])
    com_clean = clean_com(row["com"])
    if pd.notna(dep_clean) and pd.notna(com_clean):
        return dep_clean + com_clean
    return np.nan

merged_df["insee_code"] = merged_df.apply(build_insee, axis=1)

print(merged_df[["dep", "com", "insee_code"]].head())

paris_fix = {f"750{str(i).zfill(2)}": f"751{str(i).zfill(2)}" for i in range(1, 21)}
merged_df["insee_code"] = merged_df["insee_code"].replace(paris_fix)
centroids = pd.read_csv(
    "../data/raw/20230823-communes-departement-region.csv",
    sep=",", encoding="utf-8",
    usecols=["code_commune_INSEE", "latitude", "longitude"]
)

centroids.rename(columns={
    "code_commune_INSEE": "insee_code",
    "latitude": "lat_centroid",
    "longitude": "long_centroid"
}, inplace=True)

centroids["insee_code"] = centroids["insee_code"].astype(str).str.zfill(5)

merged_df = merged_df.merge(centroids, how="left", on="insee_code")

merged_df["gps_combined"] = merged_df["gps_combined"].astype(str).replace("nan", np.nan)

def choose_best_gps(row):
    if pd.notna(row.get("gps_combined")) and row["gps_combined"] != "nan":
        return row["gps_combined"]
    elif pd.notna(row.get("lat_centroid")) and pd.notna(row.get("long_centroid")):
        return f"{row['lat_centroid']},{row['long_centroid']}"
    else:
        return np.nan

merged_df["gps_combined_final"] = merged_df.apply(choose_best_gps, axis=1)

coverage = merged_df["gps_combined_final"].notna().mean() * 100
print(f"✅ Final GPS coverage (original or centroid): {coverage:.2f}%")
def get_equipment_status(row, target_code):
    if not pd.isna(row.get("secu1")):
        values = [row.get(f"secu{i}") for i in range(1, 4)]
        if any(v == target_code for v in values):
            return 1
        if all(v in [-1, 8, None, np.nan] for v in values):
            return -1
        return 0
    secu_val = row.get("secu")
    if pd.notna(secu_val):
        s = str(secu_val).strip()
        if len(s) == 2 and s[0] == str(target_code):
            if s[1] == "1":
                return 1
            elif s[1] == "2":
                return 0
            elif s[1] == "3":
                return -1
    return -1
    
merged_df["belt_status"] = merged_df.apply(get_equipment_status, target_code=1, axis=1)
merged_df["helmet_status"] = merged_df.apply(get_equipment_status, target_code=2, axis=1)
merged_df["child_device_status"] = merged_df.apply(get_equipment_status, target_code=3, axis=1)
merged_df["reflective_vest_status"] = merged_df.apply(get_equipment_status, target_code=4, axis=1)
columns_to_drop = [
    "com", "adr", "gps", "lat", "long", "dep", "v1", "v2", "pr", "pr1", "lartpc",
    "env1", "occutc", "secu", "num_veh_usag", "vma", "id_vehicule", "motor",
    "id_vehicule_usag", "secu1", "secu2", "secu3", "id_usager", "lat_clean",
    "long_clean", "gps_combined", "lat_centroid", "long_centroid"
]

merged_df = merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns])

print("Remaining columns:", merged_df.columns.tolist())
output_path = "../data/processed/accidents_processed.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged_df.to_csv(output_path, index=False)

print(f"✅ Processed data saved to: {output_path}")


file_path = "../data/processed/accidents_processed.csv"
df = pd.read_csv(file_path, low_memory=False)

df["age"] = df["an"] - df["an_nais"]
df["larrout_numeric"] = pd.to_numeric(df["larrout"], errors="coerce")
df["nbv"] = pd.to_numeric(df["nbv"], errors="coerce").astype("Int64")

variables = ["hour", "age", "nbv", "larrout_numeric"]
summary_stats = df[variables].describe().T
print(summary_stats)

df_cleaned = df.copy()
df_cleaned.loc[(df_cleaned["nbv"] < 1) | (df_cleaned["nbv"] > 10), "nbv"] = pd.NA
df_cleaned.loc[df["larrout_numeric"] < 1, "larrout_numeric"] = pd.NA
df_cleaned[["lat_clean", "long_clean"]] = df_cleaned["gps_combined_final"].str.split(",", expand=True).astype(float)

df_cleaned["insee_code"] = df_cleaned["insee_code"].astype(str)
df_cleaned["insee_prefix"] = df_cleaned["insee_code"].str[:2]

mainland_prefixes = [str(i).zfill(2) for i in range(1, 96)] + ["2A", "2B"]
overseas_prefixes = ["97", "98"]

df_cleaned["gps_insee_mismatch"] = False
df_cleaned.loc[
    (df_cleaned["insee_prefix"].isin(mainland_prefixes)) &
    (~df_cleaned["lat_clean"].between(41, 52) | ~df_cleaned["long_clean"].between(-5, 10)),
    "gps_insee_mismatch"
] = True
df_cleaned.loc[
    (df_cleaned["insee_prefix"].isin(overseas_prefixes)) &
    (df_cleaned["lat_clean"].between(41, 52) & df_cleaned["long_clean"].between(-5, 10)),
    "gps_insee_mismatch"
] = True

total_rows = len(df_cleaned)
mismatch_count = df_cleaned["gps_insee_mismatch"].sum()
mismatch_percent = round((mismatch_count / total_rows) * 100, 2)
print(f"Total rows: {total_rows}")
print(f"GPS–INSEE mismatches: {mismatch_count}")
print(f"Percentage of mismatches: {mismatch_percent}%")
df_cleaned.loc[df_cleaned["gps_insee_mismatch"], ["lat_clean", "long_clean"]] = pd.NA

categorical_cleaning_rules = {
    "lum": [1, 2, 3, 4, 5],
    "agg": [1, 2],
    "int": list(range(1, 10)),
    "atm": list(range(1, 10)),
    "col": list(range(1, 8)),
    "catr": [1, 2, 3, 4, 5, 6, 7, 9],
    "circ": [1, 2, 3, 4],
    "vosp": [0, 1, 2, 3],
    "prof": [1, 2, 3, 4],
    "plan": [1, 2, 3, 4],
    "surf": list(range(1, 10)),
    "infra": list(range(0, 10)),
    "situ": [0, 1, 2, 3, 4, 5, 6, 8],
    "senc": [0, 1, 2, 3],
    "obs": list(range(0, 10)),
    "obsm": list(range(0, 5)),
    "choc": list(range(0, 9)),
    "manv": list(range(1, 14)),
    "place": list(range(1, 11)),
    "catu": [1, 2, 3],
    "grav": [1, 2, 3, 4],
    "sexe": [1, 2],
    "trajet": [1, 2, 3, 4, 5, 9],
    "locp": list(range(1, 8)),
    "actp": list(range(1, 10)),
    "etatp": [1, 2, 3],
    "catv": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,30,31,32,33,34,35,36,37,38,39,40,41,42,43,50,60,80,99],
    "belt_status": [0, 1],
    "helmet_status": [0, 1],
    "child_device_status": [0, 1],
    "reflective_vest_status": [0, 1]
}

for col, valid_values in categorical_cleaning_rules.items():
    if col in df_cleaned.columns:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
        df_cleaned.loc[df_cleaned[col] == -1, col] = pd.NA
        df_cleaned.loc[~df_cleaned[col].isin(valid_values), col] = pd.NA

outlier_counts = {}
for col, valid_values in categorical_cleaning_rules.items():
    if col in df_cleaned.columns:
        original_col = pd.to_numeric(df_cleaned[col], errors="coerce")
        invalid_mask = ~original_col.isin(valid_values)
        outlier_count = invalid_mask.sum()
        outlier_counts[col] = outlier_count

for col, count in outlier_counts.items():
    print(f"{col}: {count} invalid values replaced with NaN")

df_cleaned = df_cleaned.dropna(subset=["grav"])

def count_missing_all_forms_with_percent(series):
    total = len(series)
    missing = (series.isna().sum() + series.apply(lambda x: isinstance(x, str) and x.strip() == "").sum())
    percent = (missing / total * 100) if total > 0 else 0
    return missing, round(percent, 2)

missing_data = {"column": [], "missing_count": [], "missing_percent": []}
for col in df_cleaned.columns:
    count, percent = count_missing_all_forms_with_percent(df_cleaned[col])
    missing_data["column"].append(col)
    missing_data["missing_count"].append(count)
    missing_data["missing_percent"].append(percent)
missing_summary = pd.DataFrame(missing_data).sort_values(by="missing_percent", ascending=False).reset_index(drop=True)
print(missing_summary)

columns_to_drop = ["locp", "actp", "etatp", "reflective_vest_status", "child_device_status", "helmet_status", "larrout_numeric",
                   "voie", "larrout", "gps_combined_final", "an_nais", "insee_prefix", "weekday_name", "num_veh", "hrmn", "gps_insee_mismatch",
                   "an", "mois", "jour"]
df_cleaned = df_cleaned.drop(columns=columns_to_drop, errors="ignore")

categorical_columns = ["lum","agg","int","atm","col","catr","circ","vosp","prof","plan","surf","infra","situ","senc","catv","obs","obsm","choc","manv","place","catu","grav","sexe","trajet","belt_status"]
for col in categorical_columns:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].astype("category")

df_cleaned["datetime"] = pd.to_datetime(df_cleaned["datetime"], errors="coerce")
df_cleaned["age"] = df_cleaned["age"].astype("Int64")
print(df_cleaned.dtypes)

file_path = "../data/processed/accidents_cleaned.csv"
df_cleaned.to_csv(file_path, index=False)

file_path = "../data/processed/accidents_cleaned.csv"
df = pd.read_csv(file_path, low_memory=False)

df.drop(columns=['num_acc', 'hrmn_clean', 'insee_code', 'lat_clean', 'long_clean', 'manv', 'trajet'], inplace=True, errors='ignore')
df['belt_status'] = df['belt_status'].fillna(-1)
df.dropna(inplace=True)

categorical_columns = ["lum","agg","int","atm","col","catr","circ","vosp","prof","plan","surf","infra","situ","senc","catv","obs","obsm","choc","place","catu","grav","sexe","belt_status","dayofweek"]
for col in categorical_columns:
    if col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("Int64")
        df[col] = df[col].astype(str).fillna("unknown").astype("category")

df["age"] = df["age"].astype("Int64")
df["nbv"] = df["nbv"].astype("Int64")
print(df.dtypes)
print(df.head())

place_grav_ct = pd.crosstab(df['place'], df['grav'])
place_grav_colnorm = place_grav_ct.div(place_grav_ct.sum(axis=0), axis=1)
plt.figure(figsize=(6, 4))
sns.heatmap(place_grav_colnorm, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Proportion'})
plt.title('Proportion of Place by Injury Severity (grav)')
plt.xlabel('grav (Injury Severity)')
plt.ylabel('place (Seat/Position)')
plt.tight_layout()
plt.show()

df['agg_catr'] = df['agg'].astype(str) + "_" + df['catr'].astype(str)
df['agg_catr'] = df['agg_catr'].astype('category')
print(df['agg_catr'].value_counts())

df['catv'] = df['catv'].astype(int)
group_definitions = {1:[1,30,31,32,33,34,35,36,50,60,80],2:[7,10],3:[13,14,15,16,17],4:[37,38],5:[20,21,99],6:[39,40],7:[41,42,43],8:[0,2,3,4,5,6,8,9,11,12,18,19]}
code_to_group = {code: group for group, codes in group_definitions.items() for code in codes}
df['catv_group'] = df['catv'].map(code_to_group).fillna(9).astype(int)
print(df['catv_group'].value_counts().sort_index())

df['rush_hour'] = df['hour'].apply(lambda h: 1 if h in [7,8,16,17,18] else 0)
print(df['rush_hour'].value_counts().sort_index())

df['month'] = pd.to_datetime(df['datetime']).dt.month
def month_to_season(month):
    if month in [3,4,5]:
        return 1
    elif month in [6,7,8]:
        return 2
    elif month in [9,10,11]:
        return 3
    else:
        return 4
df['season'] = df['month'].apply(month_to_season).astype('category')
print(df['season'].value_counts().sort_index())

age_bins = [0,17,25,40,60,120]
age_labels = [1,2,3,4,5]
df['age_bin'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)
print(df['age_bin'].value_counts().sort_index())

def encode_belt_user_type(row):
    catu = int(row['catu'])
    belt = int(row['belt_status'])
    if catu == 1:
        return 1 if belt == 1 else 2
    elif catu == 2:
        return 3 if belt == 1 else 4
    elif catu == 3:
        return 5
    else:
        return 0
df['belt_user_type_code'] = df.apply(encode_belt_user_type, axis=1).astype('int8')
print(df['belt_user_type_code'].value_counts().sort_index())

categorical_additional = ['rush_hour','season','age_bin','belt_user_type_code','catv_group']
for col in categorical_additional:
    if col in df.columns:
        df[col] = df[col].astype('category')

df.drop(columns=['catv','agg','catr','place','datetime','month'], inplace=True)
print(df.dtypes)
print(df.head())
print(df.info())

df_ml = df.copy()
file_path = "../data/processed/df_for_ml.csv"
os.makedirs(os.path.dirname(file_path), exist_ok=True)
df_ml.to_csv(file_path, index=False)
print("✅ Export complete:", df_ml.shape, "→", file_path)

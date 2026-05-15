import json
import glob
import os
import subprocess

v2_nbs = sorted(glob.glob("notebooks_v2/*.ipynb"))

for nb_path in v2_nbs:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            src = ''.join(cell['source'])
            new_src = src
            
            # Change models/ to models_v2/
            if "'models/" in new_src or '"models/' in new_src or 'models/' in new_src:
                new_src = new_src.replace("'models/", "'models_v2/").replace('"models/', '"models_v2/')
                
            # Drop employee_id before saving in 05
            if '05_data_cleaning_v2' in nb_path and 'to_csv' in new_src and 'employee_performance_clean.csv' in new_src:
                if 'drop' not in new_src:
                    new_src = "if 'employee_id' in df.columns:\n    df = df.drop(columns=['employee_id'])\n" + new_src
                    
            # Remove SMOTE in 08
            if '08_pipeline_v2' in nb_path:
                if 'from imblearn.pipeline import Pipeline' in new_src:
                    new_src = new_src.replace('from imblearn.pipeline import Pipeline', 'from sklearn.pipeline import Pipeline')
                if 'SMOTE' in new_src and 'preprocessor' in new_src:
                    new_src = new_src.replace('("smote", SMOTE(random_state=42)),', '')
                    
            # Add class_weight to baselines in 09
            if '09_baseline_models_v2' in nb_path:
                if 'RandomForestClassifier(' in new_src and 'class_weight' not in new_src:
                    new_src = new_src.replace('RandomForestClassifier(random_state=42)', "RandomForestClassifier(random_state=42, class_weight='balanced')")
                if 'LogisticRegression(' in new_src and 'class_weight' not in new_src:
                    new_src = new_src.replace('LogisticRegression(max_iter=1000)', "LogisticRegression(max_iter=1000, class_weight='balanced')")

            if new_src != src:
                if isinstance(cell['source'], list):
                    cell['source'] = [line + ('\n' if i < len(new_src.split('\n')) - 1 and not line.endswith('\n') else '') for i, line in enumerate(new_src.split('\n'))]
                    # Simple hack: just write it as a single string to avoid line breaking mess
                    cell['source'] = [new_src]
                else:
                    cell['source'] = new_src
                modified = True
                
    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Modificado: {nb_path}")

# Ejecutar los notebooks modificados
for nb_path in v2_nbs:
    print(f"Ejecutando: {nb_path}...")
    try:
        subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], check=True)
        print(f"Completado: {nb_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando {nb_path}: {e}")
        break

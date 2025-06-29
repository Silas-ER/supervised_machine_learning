import pandas as pd
import requests
from pathlib import Path
import os
import kagglehub
import shutil

def download_from_kaggle(dataset_name="usdot/flight-delays"):
    """Download dataset do Kaggle e move para pasta data"""
    try:
        # Download do Kaggle
        print(f"Baixando dataset {dataset_name} do Kaggle...")
        kaggle_path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset baixado em: {kaggle_path}")
        
        # Definir pasta data do projeto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        
        # Criar pasta data se não existir
        os.makedirs(data_dir, exist_ok=True)
        
        # Mover arquivos CSV para pasta data
        kaggle_files = os.listdir(kaggle_path)
        csv_files = [f for f in kaggle_files if f.endswith('.csv')]
        
        for csv_file in csv_files:
            src = os.path.join(kaggle_path, csv_file)
            dst = os.path.join(data_dir, csv_file)
            
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"Arquivo copiado: {csv_file}")
            else:
                print(f"Arquivo já existe: {csv_file}")
        
        return data_dir
    
    except Exception as e:
        print(f"Erro ao baixar do Kaggle: {e}")
        return None
                
def load_data(dir, filename):
    """
    Carrega dados do arquivo CSV.
    Se o arquivo não existir, baixa automaticamente do Kaggle.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, dir, filename)
    
    # Se arquivo não existe, baixar do Kaggle
    if not os.path.exists(file_path):
        print(f"Arquivo {filename} não encontrado. Baixando do Kaggle...")
        result = download_from_kaggle()
        
        if result is None:
            raise FileNotFoundError(f"Não foi possível baixar o arquivo {filename}")
    
    # Verificar se arquivo existe após download
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo {filename} ainda não encontrado após download")
    
    df = pd.read_csv(file_path, low_memory=False)
    return df
import pandas as pd

# Caminho para um maior conjunto de dados de sentimentos por texto
dataset_path = 'arquivo\\baixado\\por\\voce'



try:
    data = pd.read_csv(dataset_path)
    print(data.head())
except FileNotFoundError:
    print(f"Arquivo '{dataset_path}' não encontrado.")
except pd.errors.ParserError as e:
    print(f"Erro ao analisar o arquivo CSV: {e}")

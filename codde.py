import pandas as pd
def categoria_genero(valor):
    if valor == 0:
        return "Masculino"
    elif valor == 1:
        return "Feminino"
    else:
        return "Outro"
# Carregar o dataset
df = pd.read_csv("SCsv.csv")

# Selecionar a primeira coluna (Gênero) e as últimas 5 colunas
first_column = df.iloc[:, 0]  # Gênero
last_5_columns = df.iloc[:, -5:]  # Últimas 5 colunas de interesse (respostas)
# Concatenar a primeira coluna com as últimas 5 colunas
df_num = pd.concat([first_column, last_5_columns], axis=1)
df_num['Genero'] = df_num['Genero'].apply(categoria_genero)

# Agrupar por Gênero e calcular as somas das respostas
contagem = df_num.groupby(['Genero']).sum()
# Calcular o total de pessoas por gênero
total_por_genero = df_num['Genero'].value_counts()
# Calcular a percentagem (dividindo pela soma por gênero)
percentagem = (contagem.T / total_por_genero) * 100
# Exibir a tabela de percentagens
print(percentagem)
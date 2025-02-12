import pandas as pd

def categoria_opiniao(valor):
    if valor == 0:
        return 0
    elif 1 <= valor <= 4:
        return 1
    elif 5 <= valor <= 6:
        return 2
    else:
        return 3

def classificar(row):
    max_col = row.idxmax()  # Encontra a coluna com o maior valor
    return row == row[max_col]  # Marca 1 na coluna com maior valor e 0 nas outras


# Load the CSV with the correct delimiter
df = pd.read_csv("FinalCsv.csv")

# Apply the classification logic only if the column contains numeric values
df['Opiniao Politica'] = df['Opiniao Politica'].apply(categoria_opiniao)

#Apply fillna to the missing values
df['Estado civil'] = df['Estado civil'].fillna(0)
df['Grau de escolaridade'] = df['Grau de escolaridade'].fillna(0)
df['Situação profissional'] = df['Situação profissional'].fillna(0)

#Selecting the columns
df_numeric = df.iloc[:, 6:]
#Applying the function to said columns
df_numeric = df_numeric.apply(classificar, axis=1).astype(int)
#Adding the values to the original dataframe
df[df_numeric.columns] = df_numeric

new_df = df  # Select specific columns

# Save to a new CSV file
new_df.to_csv('SCsv.csv', index=False)

import pandas as pd

# Load the CSV with the correct delimiter
old_csv = pd.read_csv("saude.csv", delimiter=";")

#Renaming the column to Opinião Política
old_column_name = old_csv.columns[8]
old_csv.rename(columns={old_column_name: "Opiniao Politica"}, inplace=True)

# Fill NaN values in the column and assign it back
old_csv['Opiniao Politica'] = old_csv['Opiniao Politica'].fillna(0)
old_csv['Opiniao Politica'] = old_csv['Opiniao Politica'].fillna(0)

#Drop the columns that don´t matter
old_csv.drop(columns=old_csv.columns[[1,5,6,7]], inplace=True)
old_csv.drop(columns=old_csv.columns[5:74], inplace=True)

#Renaming the column 'Idade Geração' to Idade
old_column_name = old_csv.columns[5]
old_csv.rename(columns={old_column_name: "Idade"}, inplace=True)

#Drop the remaining columns
old_csv.drop(columns=old_csv.columns[6:25], inplace=True)

#Moving Age column to the 2nd position
column_to_move = old_csv.columns[5]  # Column at index 6
column_data = old_csv.pop(column_to_move)  # Remove it from its original place
old_csv.insert(1, column_to_move, column_data)  # Insert at index 2
"""
# Modify data if needed (select specific columns, filter rows, etc.)
new_df = old_csv  # Select specific columns

# Save to a new CSV file
new_df.to_csv('new_file.csv', index=False)
"""

#Change the new csv to the correct names and creating a new csv
df = pd.read_csv("new_file.csv")

old_column_name = df.columns[4]
df.rename(columns={old_column_name: "Situação profissional"}, inplace=True)

old_column_name = df.columns[6]
df.rename(columns={old_column_name: "Espontaneo"}, inplace=True)

old_column_name = df.columns[7]
df.rename(columns={old_column_name: "Intuitivo"}, inplace=True)

old_column_name = df.columns[8]
df.rename(columns={old_column_name: "Dependente"}, inplace=True)

old_column_name = df.columns[9]
df.rename(columns={old_column_name: "Evitante"}, inplace=True)

old_column_name = df.columns[10]
df.rename(columns={old_column_name: "Racional"}, inplace=True)

new_df = df  # Select specific columns

# Save to a new CSV file
new_df.to_csv('FinalCsv.csv', index=False)

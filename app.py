import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay,confusion_matrix, accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

pd.set_option('Display.max_columns',500)
pd.set_option('Display.max_rows',500)
def categoria_genero(valor):
    if valor == 0:
        return "Masculino"
    elif valor == 1:
        return "Feminino"
    else:
        return "Outro"

def categoria_ID(valor):
    if valor == 1:
        return "Geração Z"
    elif valor == 2:
        return "Geração Y"
    elif valor == 3:
        return "Geração X"
    else:
        return "Geração BB"

def categoria_SP(valor):
    if valor == 1:
        return "Emprgado(a)"
    elif valor == 2:
        return " Desempregado(a)"
    elif valor == 3:
        return "Estudante"
    elif valor == 4:
        return "Trabalhador Estudante"
    elif valor == 5:
        return "Reformado(a)"
    else:
        return "Outra"
def categoria_EC(valor):
    if valor == 1:
        return "Solteiro"
    elif valor == 2:
        return "Casado/União Facto"
    elif valor == 3:
        return "Divorciado/Separado"
    elif valor == 4:
        return "Viúvo"
    else:
        return "Outro"

def categoria_Esc(valor):
    if valor == 1:
        return "1º Ciclo"
    elif valor == 2:
        return "2º Ciclo"
    elif valor == 3:
        return "3º Ciclo"
    elif valor == 4:
        return "Ensino Secundário"
    elif valor == 5:
        return "Bacharelato"
    elif valor == 6:
        return "Licenciatura"
    elif valor == 7:
        return "Mestrado"
    elif valor == 8:
        return "Doutoramento"
    else:
        return "Outro"

def categoria_opiniao(valor):
    if valor == 0:
        return "Sem Opinião Política"
    elif valor == 1:
        return "Esquerda"
    elif valor == 2:
        return "Centro"
    else:
        return "Direita"

def previsões(valor,df):
    if valor == 1:
        df = df.dropna()
        dfX = df.drop(['Racional'], axis=1)
        dfY = df['Racional']
        X_train, X_test, Y_train, Y_test = train_test_split(dfX, dfY, test_size=0.3)
        clf = GaussianNB()
        clf.fit(X_train, Y_train)
        nbScore = round((clf.score(X_test, Y_test)) * 100, 4)
        print('Score of the Classification is: {}'.format(nbScore))
        y_pred = clf.predict(X_test)
        print(classification_report(Y_test, y_pred))

        class_names = ['No', 'Yes']
        cm = confusion_matrix(Y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        plt.show()

        clf = GaussianNB()
        scores = cross_val_score(clf, dfX, dfY, cv=5)
        print(scores)
    elif valor == 2:
        # Step 1: Feature Selection
        X = df.drop(['Espontaneo', 'Intuitivo', 'Dependente', 'Evitante', 'Racional'], axis=1)
        y = df[['Espontaneo', 'Intuitivo', 'Dependente', 'Evitante', 'Racional']]

        # Step 2: Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        #Feature Scaling - estandardiza a amostra
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #Train XGBoost with MultiOutputClassifier
        xgb_model = MultiOutputClassifier(XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6))
        xgb_model.fit(X_train, y_train)

        #Decide se uma amostra é ou não predita (se prob da amostra >= treshold)
        y_pred_proba = xgb_model.predict_proba(X_test)  # Get probabilities
        threshold = 0.4
        y_pred_adjusted = np.array(
            [probs[:, 1] for probs in y_pred_proba]).T  # Pegamos a probabilidade da classe positiva
        y_pred_adjusted = (y_pred_adjusted >= threshold).astype(int)

        #Avaliar Performance
        print(f'Optimized Accuracy: {accuracy_score(y_test, y_pred_adjusted)}')
        print(f'Report: \n{classification_report(y_test, y_pred_adjusted, zero_division=1)}')
        # Exibir o nº de amostras sem previsão
        no_pred_samples = (y_pred_adjusted.sum(axis=1) == 0).sum()
        print(f"Número de amostras sem previsão: {no_pred_samples} de {len(y_test)}")
        #Matriz de Confusão
        class_names = ['Espontaneo', 'Intuitivo', 'Dependente', 'Evitante', 'Racional']
        cm = multilabel_confusion_matrix(y_test, y_pred_adjusted)
        for i, label in enumerate(class_names):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm[i], display_labels=[f'Not {label}', label])
            disp.plot()
            plt.title(f'Confusion Matrix for {label}')
            plt.show()
    elif valor == 3:
        # Passo 2: Separar as variáveis independentes (features) da variável dependente (target)
        X = df.drop(['Espontaneo', 'Intuitivo', 'Dependente', 'Evitante', 'Racional'], axis=1)  # Features
        y = df[['Espontaneo', 'Intuitivo', 'Dependente', 'Evitante', 'Racional']]  # Target com múltiplas classes

        # Passo 3: Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        #Treino com o random forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2)
        multi_target_rf = MultiOutputClassifier(rf)
        multi_target_rf.fit(X_train, y_train)

        y_pred = multi_target_rf.predict(X_test)

        # Exibir a acurácia
        print(f'Acurácia: {accuracy_score(y_test, y_pred)}')

        # Exibir o relatório de classificação
        print(f'Relatório de Classificação: \n{classification_report(y_test, y_pred, zero_division=1)}')

        #Exibir o nº de amostras sem previsão
        no_pred_samples = (y_pred.sum(axis=1) == 0).sum()
        print(f"Número de amostras sem previsão: {no_pred_samples} de {len(y_test)}")

        # Passo 7: Matriz de Confusão
        class_names = ['Espontaneo', 'Intuitivo', 'Dependente', 'Evitante', 'Racional']
        cm = confusion_matrix(y_test.values.argmax(axis=1),
                              y_pred.argmax(axis=1))  # Ajuste para múltiplas classes
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        plt.show()
    elif valor == 0:
        print("")
    else:
        print("Please choose a Number beetween 0 and 2!")


def graphs(valor, df):
    if valor == 1:
        sns.heatmap(df.corr(), annot=True)
        plt.show()
    elif valor == 2:
        # Concatenating the selected columns into a new DataFrame
        df_Bar = pd.concat([df.iloc[:, 2], df.iloc[:, 4]], axis=1)
        # Apply transformations if needed
        df_Bar['Estado civil'] = df_Bar['Estado civil'].apply(categoria_EC)
        df_Bar['Situação profissional'] = df_Bar['Situação profissional'].apply(categoria_SP)
        # Create count plot to show number of people by category
        sns.histplot(data=df_Bar,
                     x='Estado civil',
                     hue='Situação profissional',  # Show density instead of count
                     common_norm=False,  # Ensure separate normalization per gender
                     multiple="dodge",  # Stack bars by gender
                     bins=8, shrink=0.8)
        # Improve plot readability
        plt.title("Distribuição da Situação Profissional por Estado Civil")
        plt.xlabel("Estado Civil")
        plt.ylabel("Número de Pessoas")
        plt.xticks(rotation=45)  # Rotate for better visibility if categories are long
        plt.show()
    elif valor == 3:
        column = df.iloc[:, 0]  # Gênero
        last_5_columns = df.iloc[:, 5]  # Estado Civil
        # Concatenar a primeira coluna com as últimas 5 colunas
        df_Bar = pd.concat([column, last_5_columns], axis=1)
        df_Bar['Genero'] = df_Bar['Genero'].apply(categoria_genero)
        sns.boxplot(x="Genero",
                    y="Opiniao Politica", data=df_Bar)
        plt.title("Distribuição da Opinião Política por Gênero")
        plt.xlabel("Gênero")
        plt.ylabel("Opinião Política")
        plt.xticks(rotation=45)  # Rotate for readability
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    elif valor == 4:
        column = df.iloc[:, 0]  # Gênero
        last_5_columns = df.iloc[:, 3]  # Estado Civil
        # Concatenar a primeira coluna com as últimas 5 colunas
        df_Bar = pd.concat([column, last_5_columns], axis=1)
        df_Bar['Genero'] = df_Bar['Genero'].apply(categoria_genero)
        df_Bar['Grau de escolaridade'] = df_Bar['Grau de escolaridade'].apply(categoria_Esc)

        sns.violinplot(x="Genero",
                       y="Grau de escolaridade", data=df_Bar)
        # Display the plot
        plt.title("Distribuição do Grau de Escolaridade por Gênero")
        plt.xlabel("Gênero")
        plt.ylabel("Grau de Escolaridade")
        plt.show()
    elif valor == 0:
        print("")
    else:
        print("Please choose a Number beetween 0 and 4!")

def analysis(valor,df):
    if valor == 1:
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
        percentagem = round((contagem.T / total_por_genero) * 100, 2)
        # Exibir a tabela de percentagens
        print(percentagem)
    elif valor == 2:
        # Selecionar a primeira coluna (Gênero) e as últimas 5 colunas
        column = df.iloc[:, 2]  # Gênero
        last_5_columns = df.iloc[:, -5:]  # Últimas 5 colunas de interesse (respostas)
        # Concatenar a primeira coluna com as últimas 5 colunas
        df_num = pd.concat([column, last_5_columns], axis=1)
        df_num['Estado civil'] = df_num['Estado civil'].apply(categoria_EC)
        # Agrupar por Gênero e calcular as somas das respostas
        contagem = df_num.groupby(['Estado civil']).sum()
        # Calcular o total de pessoas por gênero
        total_por_genero = df_num['Estado civil'].value_counts()
        # Calcular a percentagem (dividindo pela soma por gênero)
        percentagem = round((contagem.T / total_por_genero) * 100, 2)
        # Exibir a tabela de percentagens
        print(percentagem)
    elif valor == 3:
        # Selecionar a primeira coluna (Gênero) e as últimas 5 colunas
        column = df.iloc[:, 3]  # Gênero
        last_5_columns = df.iloc[:, -5:]  # Últimas 5 colunas de interesse (respostas)
        # Concatenar a primeira coluna com as últimas 5 colunas
        df_num = pd.concat([column, last_5_columns], axis=1)
        df_num['Grau de escolaridade'] = df_num['Grau de escolaridade'].apply(categoria_Esc)
        # Agrupar por Gênero e calcular as somas das respostas
        contagem = df_num.groupby(['Grau de escolaridade']).sum()
        # Calcular o total de pessoas por gênero
        total_por_genero = df_num['Grau de escolaridade'].value_counts()
        # Calcular a percentagem (dividindo pela soma por gênero)
        percentagem = round((contagem.T / total_por_genero) * 100, 2)
        # Exibir a tabela de percentagens
        print(percentagem)
    elif valor == 4:
        # Selecionar a primeira coluna (Gênero) e as últimas 5 colunas
        column = df.iloc[:, 4]  # Gênero
        last_5_columns = df.iloc[:, -5:]  # Últimas 5 colunas de interesse (respostas)
        # Concatenar a primeira coluna com as últimas 5 colunas
        df_num = pd.concat([column, last_5_columns], axis=1)
        df_num['Situação profissional'] = df_num['Situação profissional'].apply(categoria_SP)
        # Agrupar por Gênero e calcular as somas das respostas
        contagem = df_num.groupby(['Situação profissional']).sum()
        # Calcular o total de pessoas por gênero
        total_por_genero = df_num['Situação profissional'].value_counts()
        # Calcular a percentagem (dividindo pela soma por gênero)
        percentagem = round((contagem.T / total_por_genero) * 100, 2)
        # Exibir a tabela de percentagens
        print(percentagem)
    elif valor == 5:
        # Selecionar a primeira coluna (Gênero) e as últimas 5 colunas
        column = df.iloc[:, 5]  # Gênero
        last_5_columns = df.iloc[:, -5:]  # Últimas 5 colunas de interesse (respostas)
        # Concatenar a primeira coluna com as últimas 5 colunas
        df_num = pd.concat([column, last_5_columns], axis=1)
        df_num['Opiniao Politica'] = df_num['Opiniao Politica'].apply(categoria_opiniao)
        # Agrupar por Gênero e calcular as somas das respostas
        contagem = df_num.groupby(['Opiniao Politica']).sum()
        # Calcular o total de pessoas por gênero
        total_por_genero = df_num['Opiniao Politica'].value_counts()
        # Calcular a percentagem (dividindo pela soma por gênero)
        percentagem = round((contagem.T / total_por_genero) * 100, 2)
        # Exibir a tabela de percentagens
        print(percentagem)
    elif valor == 6:
        # Selecionar a primeira coluna (Gênero) e as últimas 5 colunas
        column = df.iloc[:, 1]  # Gênero
        last_5_columns = df.iloc[:, -5:]  # Últimas 5 colunas de interesse (respostas)
        # Concatenar a primeira coluna com as últimas 5 colunas
        df_num = pd.concat([column, last_5_columns], axis=1)
        df_num['Idade'] = df_num['Idade'].apply(categoria_ID)
        # Agrupar por Gênero e calcular as somas das respostas
        contagem = df_num.groupby(['Idade']).sum()
        # Calcular o total de pessoas por gênero
        total_por_genero = df_num['Idade'].value_counts()
        # Calcular a percentagem (dividindo pela soma por gênero)
        percentagem = round((contagem.T / total_por_genero) * 100, 2)
        # Exibir a tabela de percentagens
        print(percentagem)
    elif valor == 7:
        # Selecionar a primeira coluna (Gênero) e as últimas 5 colunas
        column = df.iloc[:, 0]  # Gênero
        last_5_columns = df.iloc[:, 2]  # Estado Civil
        # Concatenar a primeira coluna com as últimas 5 colunas
        df_num = pd.concat([column, last_5_columns], axis=1)
        df_num['Genero'] = df_num['Genero'].apply(categoria_genero)
        df_num['Estado civil'] = df_num['Estado civil'].apply(categoria_EC)
        # Agrupar por Gênero e Estado Civil e contar as ocorrências
        contagem = df_num.groupby(['Genero', 'Estado civil']).size().unstack(fill_value=0)
        # Calcular o total de pessoas por Gênero
        total_por_genero = df_num['Genero'].value_counts()
        # Calcular a percentagem para cada estado civil dentro de cada gênero
        percentagem_estado_civil = round((contagem.T / total_por_genero) * 100, 2)
        # Exibir a tabela de percentagens por estado civil
        print(percentagem_estado_civil)
    elif valor == 0:
        print("")
    else:
        print("Please choose a Number beetween 0 and 7!")


df = pd.read_csv("SCsv.csv")

option = -1

while option != 0:
        option = int(input("What's the Option you want ?\nOption 1- Exploratory Analysis\nOption 2- Graphic Analysis\nOption 3- Preditive Analysis\nOption 0 - End Programm\n"))
        if option == 1:
            op = -1
            while op != 0:
                op = int(input("What's the Analysis you want ?\nOption 1 - By Genre\nOption 2 - By Civil State\nOption 3 - By Scolarship\nOption 4 - By Professional Situation\nOption 5 - By Political Opinion\nOption 6 - By Age Group\nOption 7 - Analysis beetwen Genre and Civil State\nOption 0 - Return to the menu\n"))
                analysis(op, df)
        elif option == 2:
            op = -1
            while op != 0:
                op = int(input("What's the Graphic Analysis you want ?\nOption 1 - Heatmap\nOption 2 - Amount of people by every Civil State\nOption 3 -Distribution of Political Opinial by Genre\nOption 4 - Violin Plot representing the density of Education level by Genre\nOption 0 - Return to the menu\n"))
                graphs(op, df)
        elif option == 3:
            op = -1
            while op != 0:
                op = int(input("What's the Preditive Analysis you want ?\nOption 1 - Classification of Racional People\nOption 2 - MultiClassification(Low Accuracy)\nOption 2 - MultiClassification(Higher Accuracy)\nOption 0 - Return to the menu\n"))
                previsões(op, df)
else:
        print("Please choose a Number beetween 0 and 3!")

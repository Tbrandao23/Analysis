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

#Function to categorize the genre
def categoria_genero(valor):
    if valor == 0:
        return "Masculino"
    elif valor == 1:
        return "Feminino"
    else:
        return "Outro"
#Function to categorize the Age group
def categoria_ID(valor):
    if valor == 1:
        return "Geração Z"
    elif valor == 2:
        return "Geração Y"
    elif valor == 3:
        return "Geração X"
    else:
        return "Geração BB"
#Function to categorize the Professional Situation
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
#Function to categorize the Civil State
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
#Function to categorize the Scolarship degree
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
#Function to categorize the political opinion
def categoria_opiniao(valor):
    if valor == 0:
        return "Sem Opinião Política"
    elif valor == 1:
        return "Esquerda"
    elif valor == 2:
        return "Centro"
    else:
        return "Direita"
#Function to the Prediction menu
def previsões(valor,df):
    if valor == 1: # Gaussian Classification
        # Step 1: Feature Selection
        df = df.dropna()
        dfX = df.drop(['Racional'], axis=1)
        dfY = df['Racional']

        # Step 2: Train-Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(dfX, dfY, test_size=0.3)
        # Train Gaussian Classifier
        clf = GaussianNB()
        clf.fit(X_train, Y_train)
        # Evaluate Performance
        nbScore = round((clf.score(X_test, Y_test)) * 100, 4)
        print('Score of the Classification is: {}'.format(nbScore))
        y_pred = clf.predict(X_test)
        print(classification_report(Y_test, y_pred))
        # Confusion Matrix
        class_names = ['No', 'Yes']
        cm = confusion_matrix(Y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        plt.show()
        # Prediction Score
        clf = GaussianNB()
        scores = cross_val_score(clf, dfX, dfY, cv=5)
        print(scores)
    elif valor == 2:
        # Step 1: Feature Selection
        X = df.drop(['Spontaneous', 'Intuitive', 'Dependent', 'Avoidant', 'Rational'], axis=1)
        y = df[['Spontaneous', 'Intuitive', 'Dependent', 'Avoidant', 'Rational']]

        # Step 2: Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Feature Scaling - Standardizes the sample
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train XGBoost with MultiOutputClassifier
        xgb_model = MultiOutputClassifier(XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6))
        xgb_model.fit(X_train, y_train)

        # Decides whether a sample is predicted (if the probability of the sample >= threshold)
        y_pred_proba = xgb_model.predict_proba(X_test)
        threshold = 0.4
        y_pred_adjusted = np.array([probs[:, 1] for probs in y_pred_proba]).T
        y_pred_adjusted = (y_pred_adjusted >= threshold).astype(int)

        # Evaluate Performance
        print(f'Optimized Accuracy: {accuracy_score(y_test, y_pred_adjusted)}')
        print(f'Report: \n{classification_report(y_test, y_pred_adjusted, zero_division=1)}')

        # Display the number of samples without prediction
        no_pred_samples = (y_pred_adjusted.sum(axis=1) == 0).sum()
        print(f"Number of samples without prediction: {no_pred_samples} out of {len(y_test)}")

        # Confusion Matrix
        class_names = ['Spontaneous', 'Intuitive', 'Dependent', 'Avoidant', 'Rational']
        cm = multilabel_confusion_matrix(y_test, y_pred_adjusted)
        for i, label in enumerate(class_names):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm[i], display_labels=[f'Not {label}', label])
            disp.plot()
            plt.title(f'Confusion Matrix for {label}')
            plt.show()
    elif valor == 3:
        # Step 2: Separate independent variables (features) from the dependent variable (target)
        X = df.drop(['Spontaneous', 'Intuitive', 'Dependent', 'Avoidant', 'Rational'], axis=1)
        y = df[['Spontaneous', 'Intuitive', 'Dependent', 'Avoidant', 'Rational']]

        # Step 3: Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Train with Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2)
        multi_target_rf = MultiOutputClassifier(rf)
        multi_target_rf.fit(X_train, y_train)

        y_pred = multi_target_rf.predict(X_test)

        # Display accuracy
        print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

        # Display classification report
        print(f'Classification Report: \n{classification_report(y_test, y_pred, zero_division=1)}')

        # Display number of samples without prediction
        no_pred_samples = (y_pred.sum(axis=1) == 0).sum()
        print(f"Number of samples without prediction: {no_pred_samples} out of {len(y_test)}")

        # Step 7: Confusion Matrix
        class_names = ['Spontaneous', 'Intuitive', 'Dependent', 'Avoidant', 'Rational']
        cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        plt.show()
    elif valor == 0:
        print("")
    else:
        print("Please choose a number between 0 and 2!")
#Function to the Graphs menu
def graphs(valor, df):
        if valor == 1:
            # Display a heatmap to show the correlation between all variables
            sns.heatmap(df.corr(), annot=True)
            plt.show()

        elif valor == 2:
            # Concatenate the selected columns into a new DataFrame
            df_Bar = pd.concat([df.iloc[:, 2], df.iloc[:, 4]], axis=1)
            # Apply transformations if needed
            df_Bar['Marital Status'] = df_Bar['Estado civil'].apply(categoria_EC)
            df_Bar['Professional Situation'] = df_Bar['Situação profissional'].apply(categoria_SP)
            # Create a histogram to show the number of people by category
            sns.histplot(data=df_Bar,
                         x='Marital Status',
                         hue='Professional Situation',  # Differentiate by Professional Situation
                         common_norm=False,  # Ensure separate normalization per category
                         multiple="dodge",  # Separate bars by category
                         bins=8, shrink=0.8)
            # Improve plot readability
            plt.title("Distribution of Professional Situation by Marital Status")
            plt.xlabel("Marital Status")
            plt.ylabel("Number of People")
            plt.xticks(rotation=45)  # Rotate for better visibility if categories are long
            plt.show()

        elif valor == 3:
            column = df.iloc[:, 0]  # Gender
            last_5_columns = df.iloc[:, 5]  # Political Opinion
            # Concatenate the selected column with the last 5 columns
            df_Bar = pd.concat([column, last_5_columns], axis=1)
            df_Bar['Gender'] = df_Bar['Genero'].apply(categoria_genero)
            sns.boxplot(x="Gender",
                        y="Opiniao Politica", data=df_Bar)
            # Display the plot
            plt.title("Distribution of Political Opinion by Gender")
            plt.xlabel("Gender")
            plt.ylabel("Political Opinion")
            plt.xticks(rotation=45)  # Rotate for readability
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()

        elif valor == 4:
            column = df.iloc[:, 0]  # Gender
            last_5_columns = df.iloc[:, 3]  # Educational Level
            # Concatenate the selected column with the last 5 columns
            df_Bar = pd.concat([column, last_5_columns], axis=1)
            df_Bar['Gender'] = df_Bar['Genero'].apply(categoria_genero)
            df_Bar['Educational Level'] = df_Bar['Grau de escolaridade'].apply(categoria_Esc)

            sns.violinplot(x="Gender",
                           y="Educational Level", data=df_Bar)
            # Display the plot
            plt.title("Distribution of Educational Level by Gender")
            plt.xlabel("Gender")
            plt.ylabel("Educational Level")
            plt.show()

        elif valor == 0:
            print("")

        else:
            print("Please choose a number between 0 and 4!")


#Function to the Analysis menu
def analysis(valor, df):
    if valor == 1:
        # Select the first column (Gender) and the last 5 columns
        first_column = df.iloc[:, 0]  # Gender
        last_5_columns = df.iloc[:, -5:]  # Last 5 columns of interest (responses)
        # Concatenate the first column with the last 5 columns
        df_num = pd.concat([first_column, last_5_columns], axis=1)
        df_num['Genero'] = df_num['Genero'].apply(categoria_genero)
        # Group by Gender and calculate the sum of the responses
        count = df_num.groupby(['Genero']).sum()
        # Calculate the total number of people by Gender
        total_by_gender = df_num['Genero'].value_counts()
        # Calculate the percentage (dividing by the total by gender)
        percentage = round((count.T / total_by_gender) * 100, 2)
        # Display the percentage table
        print(percentage)

    elif valor == 2:
        # Select the column (Marital Status) and the last 5 columns
        column = df.iloc[:, 2]  # Marital Status
        last_5_columns = df.iloc[:, -5:]  # Last 5 columns of interest (responses)
        # Concatenate the selected column with the last 5 columns
        df_num = pd.concat([column, last_5_columns], axis=1)
        df_num['Estado civil'] = df_num['Estado civil'].apply(categoria_EC)
        # Group by Marital Status and calculate the sum of the responses
        count = df_num.groupby(['Estado civil']).sum()
        # Calculate the total number of people by Marital Status
        total_by_status = df_num['Estado civil'].value_counts()
        # Calculate the percentage (dividing by the total by status)
        percentage = round((count.T / total_by_status) * 100, 2)
        # Display the percentage table
        print(percentage)

    elif valor == 3:
        # Select the column (Educational Level) and the last 5 columns
        column = df.iloc[:, 3]  # Educational Level
        last_5_columns = df.iloc[:, -5:]  # Last 5 columns of interest (responses)
        # Concatenate the selected column with the last 5 columns
        df_num = pd.concat([column, last_5_columns], axis=1)
        df_num['Grau de escolaridade'] = df_num['Grau de escolaridade'].apply(categoria_Esc)
        # Group by Educational Level and calculate the sum of the responses
        count = df_num.groupby(['Grau de escolaridade']).sum()
        # Calculate the total number of people by Educational Level
        total_by_level = df_num['Grau de escolaridade'].value_counts()
        # Calculate the percentage (dividing by the total by level)
        percentage = round((count.T / total_by_level) * 100, 2)
        # Display the percentage table
        print(percentage)

    elif valor == 4:
        # Select the column (Professional Situation) and the last 5 columns
        column = df.iloc[:, 4]  # Professional Situation
        last_5_columns = df.iloc[:, -5:]  # Last 5 columns of interest (responses)
        # Concatenate the selected column with the last 5 columns
        df_num = pd.concat([column, last_5_columns], axis=1)
        df_num['Situação profissional'] = df_num['Situação profissional'].apply(categoria_SP)
        # Group by Professional Situation and calculate the sum of the responses
        count = df_num.groupby(['Situação profissional']).sum()
        # Calculate the total number of people by Professional Situation
        total_by_situation = df_num['Situação profissional'].value_counts()
        # Calculate the percentage (dividing by the total by situation)
        percentage = round((count.T / total_by_situation) * 100, 2)
        # Display the percentage table
        print(percentage)

    elif valor == 5:
        # Select the column (Political Opinion) and the last 5 columns
        column = df.iloc[:, 5]  # Political Opinion
        last_5_columns = df.iloc[:, -5:]  # Last 5 columns of interest (responses)
        # Concatenate the selected column with the last 5 columns
        df_num = pd.concat([column, last_5_columns], axis=1)
        df_num['Opiniao Politica'] = df_num['Opiniao Politica'].apply(categoria_opiniao)
        # Group by Political Opinion and calculate the sum of the responses
        count = df_num.groupby(['Opiniao Politica']).sum()
        # Calculate the total number of people by Political Opinion
        total_by_opinion = df_num['Opiniao Politica'].value_counts()
        # Calculate the percentage (dividing by the total by opinion)
        percentage = round((count.T / total_by_opinion) * 100, 2)
        # Display the percentage table
        print(percentage)

    elif valor == 6:
        # Select the column (Age) and the last 5 columns
        column = df.iloc[:, 1]  # Age
        last_5_columns = df.iloc[:, -5:]  # Last 5 columns of interest (responses)
        # Concatenate the selected column with the last 5 columns
        df_num = pd.concat([column, last_5_columns], axis=1)
        df_num['Idade'] = df_num['Idade'].apply(categoria_ID)
        # Group by Age and calculate the sum of the responses
        count = df_num.groupby(['Idade']).sum()
        # Calculate the total number of people by Age
        total_by_age = df_num['Idade'].value_counts()
        # Calculate the percentage (dividing by the total by Age)
        percentage = round((count.T / total_by_age) * 100, 2)
        # Display the percentage table
        print(percentage)

    elif valor == 7:
        # Select the column (Gender) and the column (Marital Status)
        column = df.iloc[:, 0]  # Gender
        last_5_columns = df.iloc[:, 2]  # Marital Status
        # Concatenate the selected columns
        df_num = pd.concat([column, last_5_columns], axis=1)
        df_num['Genero'] = df_num['Genero'].apply(categoria_genero)
        df_num['Estado civil'] = df_num['Estado civil'].apply(categoria_EC)
        # Group by Gender and Marital Status and count occurrences
        count = df_num.groupby(['Genero', 'Estado civil']).size().unstack(fill_value=0)
        # Calculate the total number of people by Gender
        total_by_gender = df_num['Genero'].value_counts()
        # Calculate the percentage for each Marital Status within each Gender
        percentage_status = round((count.T / total_by_gender) * 100, 2)
        # Display the percentage table by Marital Status
        print(percentage_status)

    elif valor == 0:
        print("")

    else:
        print("Please choose a number between 0 and 7!")

#Read the csv document
df = pd.read_csv("SCsv.csv")

option = -1
#Menu selection
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

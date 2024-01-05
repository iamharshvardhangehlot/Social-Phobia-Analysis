import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os


import pickle


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv(r'anxiety.csv', encoding='latin-1')
df = pd.DataFrame(dataset)

df_skew=df.loc[:,'SPIN1':'SPIN17']
df_skew=df_skew.select_dtypes([np.int64, np.float64])
for i, col in enumerate(df_skew.columns):
    print("\nSkewness of "+col +" is", df_skew[col].skew()) #measures skewness

    # Dropping Unwanted Columns 
df = df.drop('highestleague', axis=1)
df = df.drop('Reference', axis=1)
df = df.drop('accept', axis=1)
df = df.drop('Birthplace_ISO3', axis=1)
df = df.drop('Residence_ISO3', axis=1)
df = df.drop('Timestamp', axis=1)
df = df.drop('GAD1', axis=1)
df = df.drop('GAD2', axis=1)
df = df.drop('GAD3', axis=1)
df = df.drop('GAD4', axis=1)
df = df.drop('GAD5', axis=1)
df = df.drop('GAD6', axis=1)
df = df.drop('GAD7', axis=1)
df = df.drop('GADE', axis=1)
df = df.drop('GAD_T', axis=1)
df = df.drop('SWL_T', axis=1)
df = df.drop('SPIN_T', axis=1)
df = df.drop('League', axis=1)
df = df.drop('Game', axis=1)
df = df.drop('S. No.', axis=1)
df = df.drop('Birthplace', axis=1)
df = df.drop('Residence', axis=1)
df = df.drop('SWL1', axis=1)
df = df.drop('SWL2', axis=1)
df = df.drop('SWL3', axis=1)
df = df.drop('SWL4', axis=1)
df = df.drop('SWL5', axis=1)


def remove_outliers(df, columns):
    for col in columns:
        # Identify outliers
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Mask outliers
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)

        # Remove outliers
        df.drop(df[outlier_mask].index, inplace=True)

# Specify columns to remove outliers from
columns_to_clean = ['Hours', 'streams','Age']


remove_outliers(df, columns_to_clean)
df_filtered = df
print(df_filtered)

columns_to_fill = [
    'Hours', 'streams', 'SPIN1', 'SPIN2', 'SPIN3', 'SPIN4', 'SPIN5', 'SPIN6', 'SPIN7', 'SPIN8',
    'SPIN9', 'SPIN10', 'SPIN11', 'SPIN12', 'SPIN13', 'SPIN14', 'SPIN15', 'SPIN16', 'SPIN17',
    'Narcissism'
]

# Fill missing values with median
for column in columns_to_fill:
    df_filtered[column].fillna(df_filtered[column].median(), inplace=True)
    df_filtered[column] = df_filtered[column].astype('int64')


# Replacing String Null values 
df['Work'].fillna('Unemployed', inplace=True)
df['Degree'].fillna('Nope', inplace=True)

#Data Visualization

df_filtered_box=df_filtered.loc[:,'Platform':'Playstyle']
df_filtered_box=df_filtered_box.select_dtypes([np.int64, np.float64])

df_filtered_box=df_filtered.loc[:,'Platform':'Playstyle']
df_filtered_box=df_filtered_box.select_dtypes([np.int64, np.float64])


df_filtered_box=df_filtered.loc[:,'Platform':'Playstyle']
df_filtered_box=df_filtered_box.select_dtypes([np.int64, np.float64])


# Correlation
df_filtered_corr=df_filtered.loc[:,'Platform':'Playstyle']
df_filtered_corr=df_filtered_corr.select_dtypes([np.int64, np.float64])
df_filtered_corr.corr()


#ploting the heatmap for correlation
plt.figure(figsize=(18,10))
df_filtered_corr=df_filtered_corr.corr()


# Feature Variables 
df_filtered['Fear_Component'] = df_filtered[['SPIN1', 'SPIN2', 'SPIN3', 'SPIN4', 'SPIN5', 'SPIN6']].median(axis=1).round().astype('int64')
df_filtered['Avoidance_Component'] = df_filtered[['SPIN7', 'SPIN8', 'SPIN9', 'SPIN10', 'SPIN11', 'SPIN12', 'SPIN13']].median(axis=1).round().astype('int64')
df_filtered['Physiological_Discomfort_Component'] = df_filtered[['SPIN14', 'SPIN15', 'SPIN16', 'SPIN17']].median(axis=1).round().astype('int64')


# Target Variables
df_filtered['Social_Phobia_Level'] = df_filtered[['Fear_Component', 'Avoidance_Component', 'Physiological_Discomfort_Component']].mean(axis=1).round().astype('int64')
# print(df_filtered['Social_Phobia_Level'])


# Features and target variable
features = df_filtered[['Fear_Component', 'Avoidance_Component', 'Physiological_Discomfort_Component']]
print(features)
target = df_filtered['Social_Phobia_Level']


# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random forest classifier
rf_classifier.fit(x_train, y_train)


# Accuracy Score on the training data 
x_train_prediction = rf_classifier.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)

# Accuracy score on the test data
x_test_prediction = rf_classifier.predict(x_test)
test_data_accuracy = accuracy_score(y_test,x_test_prediction)

score=cross_val_score(rf_classifier,features,target,cv=10)



#Prediction

def take_fixed_options_mcq_quiz(questions):
    # Fixed set of options for all questions
    options = ["Not At All", "A Little Bit", "Somewhat", "Very Much", "Extremely"]

    # Initialize a list to store user responses
    user_responses = []

    # Display questions and get user responses
    for i, question in enumerate(questions, start=1):
        print(f"\nQuestion {i}: {question}")
        for j, option in enumerate(options, start=1):
            print(f"{j}. {option}")
        

        # Get user input for the selected option with exception handling
        while True:
            try:
                user_input = int(input("Your answer (enter the option number): "))
                
                # Validate user input
                if 1 <= user_input <= len(options):
                    break
                else:
                    print("Invalid input. Please enter a valid option number.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

        # Store the user response
        user_responses.append(user_input)

    # Display user responses
    print("\nYour selected answers:")
    for i, response in enumerate(user_responses, start=1):
        print(f"Question {i}: {options[response - 1]}")
        
    print(user_responses)
    dd=[]

    dfs1=user_responses[:6]
    dfs2=user_responses[6:13]
    dfs3=user_responses[13:17]
    d1 = np.array(dfs1)
    d2 = np.array(dfs2)
    d3 = np.array(dfs3)

    print(d1)
    print(d2)
    print(d3)
    print("Median of Fear-Avoidance-Physiological columns")
    dd.append(np.median(d1).round().astype('int64'))
    dd.append(np.median(d2).round().astype('int64'))
    dd.append(np.median(d3).round().astype('int64'))
    # np.median(dfs)

    print(dd)
    ddd=[dd]
    # Create the numpy array 
    symptom = np.array(["none","Mild","Moderate", "High","Extreme"]) 
    
    df_val =pd.DataFrame({
        'Fear_Component': [fear_component],
        'Avoidance_Component': [avoidance_component],
        'Physiological_Discomfort_Component': [physiological_component]
    })
    # Making predictions on data
    df_predictions = rf_classifier.predict(df_val)
    print(symptom[df_predictions][0])
    
    
# Example questions (You can customize these questions)
quiz_questions = [
    "I am afraid of people in authority.",
    "I am bothered by blushing in front of people.",
    "Parties and social events scare me.",
    "I avoid talking to people I don’t know.",
    "Being criticized scares me a lot.",
    "I avoid doing things or speaking to people for fear of embarrassment.",
    "Sweating in front of people causes me distress.",
    "I avoid going to parties.",
    "I avoid activities in which I am the center of attention.",
    "Talking to strangers scares me.",
    "I avoid having to give speeches.",
    " I would do anything to avoid being criticized.",
    "Heart palpitations bother me when I am around people.",
    "I am afraid of doing things when people might be watching.",
    "Being embarrassed or looking stupid are among my worst fears.",
    "I avoid speaking to anyone in authority.",
    "Trembling or shaking in front of others is distressing to me.",
    # Add more questions as needed
]
assert len(quiz_questions) == 17
# Call the function with the example questions
take_fixed_options_mcq_quiz(quiz_questions)





# def bharat(data):
#     # Create the numpy array 
#     symptom = np.array(["none","Mild","Moderate", "High","Extreme"]) 
    
#     df_val = pd.DataFrame(data, columns=['Fear_Component', 'Avoidance_Component', 'Physiological_Discomfort_Component'])
#     # Making predictions on data
#     df_predictions = rf_classifier.predict(df_val)
# #     print(df_predictions)
# #     print(type(df_predictions))
#     print(symptom[df_predictions][0])
#     # Create a DataFrame to display the data and predictions
# #     df_with_predictions = df_val.copy()
# #     df_with_predictions['Predicted_Social_Phobia_Level'] = df_predictions
# #     print(df_with_predictions)
# #     print(symptom[df_with_predictions])
# #     print(type(df_with_predictions))
    
    
# d=[[2,2,2]]

# bharat(d)

df_filtered=df_filtered[["Fear_Component","Avoidance_Component","Physiological_Discomfort_Component","Social_Phobia_Level"]]


pickle.dump(rf_classifier, open("social_phobia_rf.pkl","wb"))











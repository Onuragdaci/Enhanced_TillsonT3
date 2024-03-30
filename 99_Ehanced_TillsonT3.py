!pip install pandas_ta
!pip install yfinance
!pip install scikit-learn
!pip install numpy
!pip install vectorbt
import pandas_ta as ta
import yfinance as yf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import numpy as np
import pandas as pd
import vectorbt as vbt
def TillsonT3(data, Length=21, vf=0.618):
    Tillson = data.copy()
    ema_first_input = (Tillson['High'] + Tillson['Low'] + 2 * Tillson['Adj Close']) / 4
    e1 = ta.ema(ema_first_input, Length)
    e2 = ta.ema(e1, Length)
    e3 = ta.ema(e2, Length)
    e4 = ta.ema(e3, Length)
    e5 = ta.ema(e4, Length)
    e6 = ta.ema(e5, Length)
    c1 = -1 * vf * vf * vf
    c2 = 3 * vf * vf + 3 * vf * vf * vf
    c3 = -6 * vf * vf - 3 * vf - 3 * vf * vf * vf
    c4 = 1 + 3 * vf + vf * vf * vf + 3 * vf * vf
    Tillson['TillsonT3'] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    Tillson = Tillson.dropna().reset_index(drop=True)
    Tillson['Entry'] = False
    Tillson['Exit'] = False
    for i in range(1, len(Tillson)):
        if Tillson.loc[i, 'TillsonT3'] > Tillson.loc[i - 1, 'TillsonT3']:
            Tillson.loc[i, 'Entry'] = True
        if Tillson.loc[i, 'TillsonT3'] < Tillson.loc[i - 1, 'TillsonT3']:
            Tillson.loc[i, 'Exit'] = True
    return Tillson

def model_selection(X, Y):
    seed = 5
    models = [('LR', LogisticRegression()),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier(n_neighbors=8)),
        ('CART', DecisionTreeClassifier()),
        ('NB', GaussianNB()),
        ('SVM', svm.SVC()),
        ('RFT', RandomForestClassifier())]

    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed) 
        cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
    
    # Find the best model based on cross-validation accuracy
    best_model_index = np.argmax(np.mean(results, axis=1))
    best_model_name = names[best_model_index]
    best_model = models[best_model_index][1]
    return best_model


df = yf.download('DOAS.IS', period='2y', interval='1d', progress=False)
df = TillsonT3(df, Length=8, vf=0.7)
df['RSI'] = ta.rsi(df['Adj Close'],timeperiod=14)
df['OBV'] = ta.obv(df['Adj Close'], df['Volume']) 
df['TSignal'] = np.where(df['Exit'], -1, np.where(df['Entry'], 1, 0))

features , target = ['RSI', 'OBV'],'TSignal'                        # Prepare features and target labels
df = df.dropna().reset_index(drop=True)
X = df[features]
y = df[target]

scaler = StandardScaler()                                                                           # Feature scaling
X_scaled = scaler.fit_transform(X)
best_model = model_selection(X_scaled, y)                                                           # Get the best model from model selection
print(best_model)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)    # Split data into training and testing sets
best_model.fit(X_train, y_train)                                                                    # Train the best model on the training set
y_pred_test = best_model.predict(X_test)                                                            # Make predictions using the best model on the testing set

test_accuracy = accuracy_score(y_test, y_pred_test)                                                 # Evaluate model performance on the testing set
print(f"Testing Accuracy: {test_accuracy:.2f}")

df_X_test = pd.DataFrame(X_test, columns=features, index=df.index[:len(X_test)])                    # Convert X_test to DataFrame for accessing index
df_test = df_X_test.copy()

df_test['Enhanced_TSignal'] = best_model.predict(X_test)
df_test['TSignal'] = df.loc[df_X_test.index, 'TSignal']                                             # Add 'TSignal' to df_test
df_test['Adj Close'] = df.loc[df_X_test.index, 'Adj Close']                                         # Add 'Close' prices to df_test

psettings = {'init_cash': 100, 'freq': 'D', 'direction': 'longonly', 'accumulate': True}
pf1 = vbt.Portfolio.from_signals(df_test['Adj Close'], entries=df_test['TSignal'] == 1, exits=df_test['TSignal'] == -1, **psettings)
pf2 = vbt.Portfolio.from_signals(df_test['Adj Close'], entries=df_test['Enhanced_TSignal'] == 1, exits=df_test['Enhanced_TSignal'] == -1, **psettings)
print(pf1.stats())
print(pf2.stats())

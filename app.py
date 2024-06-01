from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

def load_model():
    raw_df = pd.read_csv("mail_data.csv")
    dataset = raw_df.where((pd.notnull(raw_df)), '')
    dataset.loc[dataset['Category'] == 'spam', 'Category'] = 0 
    dataset.loc[dataset['Category'] == 'ham', 'Category'] = 1
    X = dataset['Message']
    y = dataset['Category']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)
    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train_features, Y_train)
    return model, feature_extraction

model, feature_extraction = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_mail = request.form['message']
        input_data_features = feature_extraction.transform([input_mail])
        prediction = model.predict(input_data_features)
        result = 'Not Spam mail' if prediction[0] == 1 else 'Spam mail'
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

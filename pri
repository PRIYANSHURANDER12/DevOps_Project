import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_excel("priyanshu.xlsx")

# Step 1: Convert Last_Trip_Rating to categories
def categorize_rating(x):
    if x <= 3:
        return 'Low'
    elif x <= 7:
        return 'Medium'
    else:
        return 'High'

df['Rating_Category'] = df['Last_Trip_Rating'].apply(categorize_rating)

# Step 2: Drop original rating column
df = df.drop(columns=['Last_Trip_Rating'])

# Step 3: Encode categorical variables
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Step 4: Train/Test split
X = df.drop(columns=['Rating_Category'])
y = df['Rating_Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict on a new sample (optional)
sample = X_test.iloc[0:1]
predicted_category = model.predict(sample)
print("Sample prediction:", predicted_category)

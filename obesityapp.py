import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('/Users/oeufveerapat/Desktop/python/AI101/Obesity prediction.csv')

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Separate features and target variable
X = df.drop('Obesity', axis=1)
y = df['Obesity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# รับ input จากผู้ใช้เพื่อพยากรณ์
print("\nกรอกข้อมูลสำหรับทำนายคลาส Obesity:")
user_input = {}

for column in X.columns:
    if column in label_encoders:  # ถ้าเป็น categorical column ที่ถูก encode
        print(f"{column} options: {list(label_encoders[column].classes_)}")
        value = input(f"Enter {column}: ")
        user_input[column] = label_encoders[column].transform([value])[0]  # แปลงเป็นเลข
    else:
        user_input[column] = float(input(f"Enter {column} (numeric): "))

# แปลง input เป็น DataFrame และ Standardize
user_df = pd.DataFrame([user_input])
user_df = scaler.transform(user_df)

# Predict
prediction = clf.predict(user_df)[0]
predicted_class = label_encoders['Obesity'].inverse_transform([prediction])[0]

print(f"\n🔮 Predicted Obesity Class: {predicted_class}")

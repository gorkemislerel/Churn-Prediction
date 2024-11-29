import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

churn = pd.read_csv("churn.csv")

df = churn.copy()

df.drop(['customerID'], axis=1, inplace=True)

dms = pd.get_dummies(df[['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                         'PaymentMethod']])

y = df['Churn']
y = y.map({'No': 0, 'Yes': 1})

X_ = df.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
              'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn',
              'MonthlyCharges', 'TotalCharges', 'tenure'], axis=1)
X = pd.concat([X_, dms], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_resampled, y_train_resampled)

y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"İlkel Accuracy Skoru: {accuracy:.2f}")

dt_params = {
    "max_depth": [6, 8],
    "min_samples_split": [10],
    "max_features": [6]
}

dt_model = DecisionTreeClassifier(random_state=42)

dt_cv_model = GridSearchCV(dt_model, 
                           dt_params, 
                           cv=10, 
                           n_jobs=1, 
                           verbose=2)
dt_cv_model.fit(X_train_resampled, y_train_resampled)

dt_tuned = DecisionTreeClassifier(**dt_cv_model.best_params_)
dt_tuned.fit(X_train_resampled, y_train_resampled)

y_pred = dt_tuned.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Tune Edilmiş Accuracy Skoru: {accuracy:.2f}")

joblib.dump(dt_tuned, 'churn_model_dt.pkl')

def predict_churn():
    try:
        gender = gender_var.get()
        senior_citizen = int(senior_citizen_var.get())
        dependents = dependents_var.get()
        phone_service = phone_service_var.get()
        multiple_lines = multiple_lines_var.get()
        internet_service = internet_service_var.get()
        online_security = online_security_var.get()
        online_backup = online_backup_var.get()
        device_protection = device_protection_var.get()
        tech_support = tech_support_var.get()
        streaming_tv = streaming_tv_var.get()
        streaming_movies = streaming_movies_var.get()
        contract = contract_var.get()
        paperless_billing = paperless_billing_var.get()
        payment_method = payment_method_var.get()

        model = joblib.load('churn_model_dt.pkl')

        user_input = pd.DataFrame({
            'SeniorCitizen': [senior_citizen],
            'gender_Female': [1 if gender == "Female" else 0],
            'gender_Male': [1 if gender == "Male" else 0],
            'Dependents_No': [1 if dependents == "No" else 0],
            'Dependents_Yes': [1 if dependents == "Yes" else 0],
            'PhoneService_No': [1 if phone_service == "No" else 0],
            'PhoneService_Yes': [1 if phone_service == "Yes" else 0],
            'MultipleLines_No': [1 if multiple_lines == "No" else 0],
            'MultipleLines_No phone service': [1 if multiple_lines == "No phone service" else 0],
            'MultipleLines_Yes': [1 if multiple_lines == "Yes" else 0],
            'InternetService_DSL': [1 if internet_service == "DSL" else 0],
            'InternetService_Fiber optic': [1 if internet_service == "Fiber optic" else 0],
            'InternetService_No': [1 if internet_service == "No" else 0],
            'OnlineSecurity_No': [1 if online_security == "No" else 0],
            'OnlineSecurity_No internet service': [1 if online_security == "No internet service" else 0],
            'OnlineSecurity_Yes': [1 if online_security == "Yes" else 0],
            'OnlineBackup_No': [1 if online_backup == "No" else 0],
            'OnlineBackup_No internet service': [1 if online_backup == "No internet service" else 0],
            'OnlineBackup_Yes': [1 if online_backup == "Yes" else 0],
            'DeviceProtection_No': [1 if device_protection == "No" else 0],
            'DeviceProtection_No internet service': [1 if device_protection == "No internet service" else 0],
            'DeviceProtection_Yes': [1 if device_protection == "Yes" else 0],
            'TechSupport_No': [1 if tech_support == "No" else 0],
            'TechSupport_No internet service': [1 if tech_support == "No internet service" else 0],
            'TechSupport_Yes': [1 if tech_support == "Yes" else 0],
            'StreamingTV_No': [1 if streaming_tv == "No" else 0],
            'StreamingTV_No internet service': [1 if streaming_tv == "No internet service" else 0],
            'StreamingTV_Yes': [1 if streaming_tv == "Yes" else 0],
            'StreamingMovies_No': [1 if streaming_movies == "No" else 0],
            'StreamingMovies_No internet service': [1 if streaming_movies == "No internet service" else 0],
            'StreamingMovies_Yes': [1 if streaming_movies == "Yes" else 0],
            'Contract_Month-to-month': [1 if contract == "Month-to-month" else 0],
            'Contract_One year': [1 if contract == "One year" else 0],
            'Contract_Two year': [1 if contract == "Two year" else 0],
            'PaperlessBilling_No': [1 if paperless_billing == "No" else 0],
            'PaperlessBilling_Yes': [1 if paperless_billing == "Yes" else 0],
            'PaymentMethod_Bank transfer (automatic)': [1 if payment_method == "Bank transfer (automatic)" else 0],
            'PaymentMethod_Credit card (automatic)': [1 if payment_method == "Credit card (automatic)" else 0],
            'PaymentMethod_Electronic check': [1 if payment_method == "Electronic check" else 0],
            'PaymentMethod_Mailed check': [1 if payment_method == "Mailed check" else 0]
        })

        prediction = model.predict(user_input)

        if prediction == 1:
            messagebox.showinfo("Tahmin Sonucu", "Churn olacak!")
        else:
            messagebox.showinfo("Tahmin Sonucu", "Churn olmayacak!")

    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu: {str(e)}")

def clear_selections():
    gender_var.set("")
    senior_citizen_var.set("")
    dependents_var.set("")
    phone_service_var.set("")
    multiple_lines_var.set("")
    internet_service_var.set("")
    online_security_var.set("")
    online_backup_var.set("")
    device_protection_var.set("")
    tech_support_var.set("")
    streaming_tv_var.set("")
    streaming_movies_var.set("")
    contract_var.set("")
    paperless_billing_var.set("")
    payment_method_var.set("")

root = tk.Tk()
root.title("Churn Tahmin Uygulaması")

canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

gender_var = tk.StringVar()
senior_citizen_var = tk.StringVar()
dependents_var = tk.StringVar()
phone_service_var = tk.StringVar()
multiple_lines_var = tk.StringVar()
internet_service_var = tk.StringVar()
online_security_var = tk.StringVar()
online_backup_var = tk.StringVar()
device_protection_var = tk.StringVar()
tech_support_var = tk.StringVar()
streaming_tv_var = tk.StringVar()
streaming_movies_var = tk.StringVar()
contract_var = tk.StringVar()
paperless_billing_var = tk.StringVar()
payment_method_var = tk.StringVar()

gender_label = tk.Label(scrollable_frame, text="Cinsiyet:")
gender_label.grid(row=0, column=0)
gender_dropdown = tk.OptionMenu(scrollable_frame, gender_var, "Female", "Male")
gender_dropdown.grid(row=0, column=1)

senior_citizen_label = tk.Label(scrollable_frame, text="Senior Citizen:")
senior_citizen_label.grid(row=1, column=0)
senior_citizen_dropdown = tk.OptionMenu(scrollable_frame, senior_citizen_var, "0", "1")
senior_citizen_dropdown.grid(row=1, column=1)

dependents_label = tk.Label(scrollable_frame, text="Dependents:")
dependents_label.grid(row=2, column=0)
dependents_dropdown = tk.OptionMenu(scrollable_frame, dependents_var, "Yes", "No")
dependents_dropdown.grid(row=2, column=1)

phone_service_label = tk.Label(scrollable_frame, text="Phone Service:")
phone_service_label.grid(row=3, column=0)
phone_service_dropdown = tk.OptionMenu(scrollable_frame, phone_service_var, "Yes", "No")
phone_service_dropdown.grid(row=3, column=1)

multiple_lines_label = tk.Label(scrollable_frame, text="Multiple Lines:")
multiple_lines_label.grid(row=4, column=0)
multiple_lines_dropdown = tk.OptionMenu(scrollable_frame, multiple_lines_var, "Yes", "No", "No phone service")
multiple_lines_dropdown.grid(row=4, column=1)

internet_service_label = tk.Label(scrollable_frame, text="Internet Service:")
internet_service_label.grid(row=5, column=0)
internet_service_dropdown = tk.OptionMenu(scrollable_frame, internet_service_var, "DSL", "Fiber optic", "No")
internet_service_dropdown.grid(row=5, column=1)

online_security_label = tk.Label(scrollable_frame, text="Online Security:")
online_security_label.grid(row=6, column=0)
online_security_dropdown = tk.OptionMenu(scrollable_frame, online_security_var, "Yes", "No", "No internet service")
online_security_dropdown.grid(row=6, column=1)

online_backup_label = tk.Label(scrollable_frame, text="Online Backup:")
online_backup_label.grid(row=7, column=0)
online_backup_dropdown = tk.OptionMenu(scrollable_frame, online_backup_var, "Yes", "No", "No internet service")
online_backup_dropdown.grid(row=7, column=1)

device_protection_label = tk.Label(scrollable_frame, text="Device Protection:")
device_protection_label.grid(row=8, column=0)
device_protection_dropdown = tk.OptionMenu(scrollable_frame, device_protection_var, "Yes", "No", "No internet service")
device_protection_dropdown.grid(row=8, column=1)

tech_support_label = tk.Label(scrollable_frame, text="Tech Support:")
tech_support_label.grid(row=9, column=0)
tech_support_dropdown = tk.OptionMenu(scrollable_frame, tech_support_var, "Yes", "No", "No internet service")
tech_support_dropdown.grid(row=9, column=1)

streaming_tv_label = tk.Label(scrollable_frame, text="Streaming TV:")
streaming_tv_label.grid(row=10, column=0)
streaming_tv_dropdown = tk.OptionMenu(scrollable_frame, streaming_tv_var, "Yes", "No", "No internet service")
streaming_tv_dropdown.grid(row=10, column=1)

streaming_movies_label = tk.Label(scrollable_frame, text="Streaming Movies:")
streaming_movies_label.grid(row=11, column=0)
streaming_movies_dropdown = tk.OptionMenu(scrollable_frame, streaming_movies_var, "Yes", "No", "No internet service")
streaming_movies_dropdown.grid(row=11, column=1)

contract_label = tk.Label(scrollable_frame, text="Contract:")
contract_label.grid(row=12, column=0)
contract_dropdown = tk.OptionMenu(scrollable_frame, contract_var, "Month-to-month", "One year", "Two year")
contract_dropdown.grid(row=12, column=1)

paperless_billing_label = tk.Label(scrollable_frame, text="Paperless Billing:")
paperless_billing_label.grid(row=13, column=0)
paperless_billing_dropdown = tk.OptionMenu(scrollable_frame, paperless_billing_var, "Yes", "No")
paperless_billing_dropdown.grid(row=13, column=1)

payment_method_label = tk.Label(scrollable_frame, text="Payment Method:")
payment_method_label.grid(row=14, column=0)
payment_method_dropdown = tk.OptionMenu(scrollable_frame, payment_method_var, "Bank transfer (automatic)", "Credit card (automatic)",
                                        "Electronic check", "Mailed check")
payment_method_dropdown.grid(row=14, column=1)

predict_button = tk.Button(root, text="Tahmin Et", command=predict_churn)
predict_button.pack(pady=10)

clear_button = tk.Button(root, text="Temizle", command=clear_selections)
clear_button.pack(pady=5)

root.mainloop()

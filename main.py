import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import joblib
from scipy.stats import rice

# Load trained model and label encoder
model = joblib.load("voltage_classifier_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Load clean dataset
data = pd.read_csv("Project_Dataset_Clean.csv")

# Step 1: Get voltage stats from stable class
stable_voltages = data[data["Label"] == "Stable"]["Voltage"]
kappa = stable_voltages.mean()
sigma = stable_voltages.std()

# Step 2: Define Pv(t) calculation function
def calculate_pv(voltage, kappa, sigma):
    if voltage == 0:
        return 1.0
    b = kappa / sigma
    lower_v, upper_v = 218.5, 241.5  # safe range
    prob_stable = rice.cdf(upper_v, b=b, scale=sigma) - rice.cdf(lower_v, b=b, scale=sigma)
    return 1 - prob_stable
def send_email_alert(voltage, label, pv):
    sender_email = "princepatel8035@gmail.com"
    receiver_email = "pcpatel8035@gmail.com"  # or someone else's
    app_password = "unxhbzihqzootagk"  # Replace with your app password (no spaces!)

    subject = "⚠️ Power Alert from AI System"
    body = f"""
    ⚠️ Voltage Alert!

    Voltage: {voltage} V
    Condition: {label}
    Instability Probability Pv(t): {round(pv * 100, 2)}%

    Please take necessary action to stabilize the system.
    """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("📧 Email alert sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# Step 3: Predict function with alerts
def predict_and_alert(voltage):
    label_num = model.predict([[voltage]])[0]
    label = encoder.inverse_transform([label_num])[0]
    pv = calculate_pv(voltage, kappa, sigma)

    print("\n🔍 Voltage Analysis")
    print(f"Voltage Input: {voltage} V")
    print(f"Predicted Label: {label}")
    print(f"Probability of Instability Pv(t): {round(pv * 100, 2)}%")

    if label == "Power Outage" or pv > 0.5:
        print("⚠️ ALERT: Voltage is unstable or dangerous!")
        send_email_alert(voltage, label, pv)
    else:
        print("✅ Status: Voltage condition is stable.\n")

# Step 4: Run prediction
# You can try changing these voltage values
test_voltages = [180, 0, 225, 245]
for voltage in test_voltages:
    predict_and_alert(voltage)

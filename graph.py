import matplotlib.pyplot as plt
import numpy as np

# הגדרת סגנון גרפי אחיד
plt.style.use("seaborn-v0_8-darkgrid")

# 📊 נתונים - אחוזי הצלחה לכל שדה מידע
fields = ["ID Number", "Case Number", "Date", "Full Name"]
success_rates = [93, 89, 83, 81]
total_cases = 100  # מספר כולל של מקרים

# 📊 נתונים - השוואה בין סימולציה לעולם האמיתי
categories = ["Simulation", "Real-World"]
success_counts = [89, 81]  # 89/100 בסימולציה, 81/100 בזמן אמת

# 🎨 צבעים מותאמים
colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

# 🔹 גרף ראשון - אחוזי הצלחה בחילוץ שדות (Success Rate of Extracted Fields)
plt.figure(figsize=(8, 5))
bars = plt.bar(fields, success_rates, color=colors, edgecolor="black", linewidth=1.2)

plt.xlabel("Extracted Fields", fontsize=12)
plt.ylabel("Success Rate (%)", fontsize=12)
plt.title("Success Rate of Extracted Fields", fontsize=14, fontweight="bold")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# הוספת תוויות בתוך העמודות (אחוזים) ומעליהן (מספר הצלחות מתוך 100)
for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval / 2, f"{yval}%", ha="center", fontsize=12, fontweight="bold", color="white")  # בתוך העמודה
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{int((success_rates[i] / 100) * total_cases)}/100", ha="center", fontsize=12, fontweight="bold")  # מעל העמודה

plt.show()

# 🔹 גרף שני - אחוזי הצלחה באפליקציה לעומת בתנאי מעבדה (Success Rate in the App vs. Lab Conditions)
plt.figure(figsize=(6, 5))
bars = plt.bar(categories, success_counts, color=["#1f77b4", "#d62728"], edgecolor="black", linewidth=1.2)

plt.xlabel("Testing Environment", fontsize=12)
plt.ylabel("Successful Extractions (per 100 cases)", fontsize=12)
plt.title("Success Rate in the App vs. Lab Conditions", fontsize=14, fontweight="bold")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# הוספת תוויות בתוך העמודות (אחוזים) ומעליהן (מספר הצלחות מתוך 100)
for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval / 2, f"{yval}%", ha="center", fontsize=12, fontweight="bold", color="white")  # בתוך העמודה
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval}/100", ha="center", fontsize=12, fontweight="bold")  # מעל העמודה

plt.show()

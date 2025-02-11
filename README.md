# 🚗 Vehicle Damage Detection App

This **AI-powered web app** allows users to **drag and drop an image of a car** to automatically detect and classify vehicle damage.

🛠 **Built with:** `Streamlit` | `PyTorch` | `ResNet50` | `Computer Vision`  

🔍 **Model detects damage based on the third-quarter front and rear views of a car** for accurate classification.

---

## 📌 Features

✅ **Drag & Drop Image Upload**  
✅ **Instant AI-Powered Damage Classification**  
✅ **6 Damage Categories**  
✅ **User-Friendly Web Interface**  

---

## 🖥️ Demo

🎥 **[Watch the Demo on YouTube](https://youtu.be/gYjGngnLAHg)**  

---

## 🎯 Model Details

- **Architecture:** Transfer learning with **ResNet50**
- **Dataset:** 2300 labeled images
- **Training Details:**
  - **Target Classes:**
    1. 🟢 **Front Normal**
    2. 🔴 **Front Crushed**
    3. 🟡 **Front Breakage**
    4. 🟢 **Rear Normal**
    5. 🔴 **Rear Crushed**
    6. 🟡 **Rear Breakage**
  - **Validation Accuracy:** ~80%

---

## 🛠 Setup Instructions

1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/vehicle-damage-detection.git
   cd vehicle-damage-detection

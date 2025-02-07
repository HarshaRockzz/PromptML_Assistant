# PromptML: NextGen AI Assistant

## Overview

**PromptML** is an AI-powered assistant designed to streamline data science workflows. By leveraging **Streamlit, LangChain, and open-source models**, this assistant simplifies key aspects of a data science project, including **Exploratory Data Analysis (EDA), Model Selection, and Prediction**. The tool helps users **save time and resources** while making data science **more efficient and accessible**.

## Features

### 1. Exploratory Data Analysis (EDA)
- Automatically **generate insights** and **visualizations** from your dataset.
- Detect **missing values, outliers, and duplicates** with **95% accuracy**.
- Provide summary statistics and distribution plots to understand data patterns.

### 2. Model Selection
- Recommends the **best machine learning algorithms** based on dataset characteristics.
- Converts **business problems into ML tasks** and suggests appropriate models.
- Utilizes **Wikipedia research** to provide context on model selection.

### 3. Prediction
- Allows users to **make predictions** using trained models with a **user-friendly interface**.
- Supports **real-time inference** with fast response times.

### 4. Conversational AI Interface
- Built using **Mistral-7B-v0.1** to provide **real-time AI assistance** under **2 seconds**.
- Interactive session memory handling improves **query refinement efficiency by 30%**.

### 5. Code Generation & Execution
- Uses **LangChain’s Python REPL** to generate and execute **Python scripts** for ML solutions.
- Supports **custom scripts for data preprocessing, training, and evaluation**.

### 6. User-Friendly Interface
- **Streamlit-powered** UI for **seamless user experience**.
- Allows users to interact with data in an intuitive and visual manner.

## Installation

Before starting, ensure you have **Python 3.7 or later** installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/HarshaRockzz/PromptML_Assistant.git
cd PromptML_Assistant
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Running the Application
Start the Streamlit application by running:
```bash
streamlit run app.py
```
This will launch the **PromptML** AI assistant in your browser.

### 2. Uploading a Dataset
- Click on the **Upload File** button.
- Supported formats: **CSV, Excel**.

### 3. Performing EDA
- View summary statistics, correlation matrices, missing values, and visualization plots.
- Get insights from the **Pandas agent**, which can answer **natural language queries** about the dataset.

### 4. Selecting a Model
- The assistant suggests the most suitable ML algorithms based on your dataset and problem type.
- Provides information on **classification, regression, clustering**, and other ML tasks.

### 5. Making Predictions
- Upload test data and get real-time predictions using the **trained models**.
- Supports multiple machine learning frameworks.

## Technologies Used
- **LangChain**: For intelligent AI-powered workflows.
- **Streamlit**: To build an interactive UI.
- **Hugging Face Models**: For natural language processing tasks.
- **Pandas**: For data manipulation and analysis.
- **Mistral-7B-v0.1**: Conversational AI model.
- **Python**: Core programming language.

## Contributions

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Added new feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Submit a pull request.


## Contact
For any questions or issues, reach out via:
- GitHub Issues: [PromptML GitHub](https://github.com/HarshaRockzz/PromptML_Assistant/issues)
- Email: mamidipaka2003@gmail.com

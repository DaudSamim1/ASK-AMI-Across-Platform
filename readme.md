# 🛠️ Flask API for Deposition Summaries & Pinecone Search

## 📖 Overview

This is a **Flask API** that provides endpoints to:

- **Store deposition summaries** in Pinecone
- **Query Pinecone for relevant summaries** using OpenAI embeddings
- **Talk to Depo API** for information
- **Validate answers** using Depo API
- **Ask Ami Agent** for information
- **Authenticate using JWT**

The API is documented using **Flasgger**, and you can access the documentation here: **/apidocs/**

---

## 🚀 Features

✅ **GraphQL Integration**: Fetches deposition summaries from an external GraphQL API.  
✅ **Pinecone Vector Search**: Stores and retrieves deposition summaries using **AI-powered embeddings**.  
✅ **OpenAI Embeddings**: Uses **text-embedding-3-small** model to create vector representations of text.  
✅ **Flask & Flasgger**: API documentation and Swagger UI support.  
✅ **JWT Authentication**: Generates and validates access tokens.  
✅ **Modal Integration**: Deploys the API using **Modal** for serverless hosting.

---

## 🏗️ Installation & Setup

### 🔹 **1. Clone the Repository**

```bash
git clone https://github.com/DaudSamim1/ASK-AMI-Across-Platform
cd ASK-AMI-Across-Platform
```

### 🔹 **2. Create a Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 🔹 **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

> **Dependencies include:**
>
> - `flask`
> - `flasgger`
> - `requests`
> - `PyJWT==2.6.0`
> - `pinecone`
> - `openai`
> - `numpy`
> - `python-dotenv`
> - `beautifulsoup4`
> - `pymongo`
> - `asgiref`
> - `werkzeug`

---

## 🔑 Environment Variables

Create a `.env` file and set up your **API keys**:

You can find a template in `.env.example`:

```ini
# Environment variable example file
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
JWT_SECRET_KEY=your-jwt-secret-key
GRAPHQL_URL=https://backend-graphql-webapp-development.up.railway.app/graphql
```

---

## ▶️ **Running the API Locally**

```bash
flask run
```

> The API will be available at:  
> **http://127.0.0.1:5000**

---

## 🔥 API Endpoints

### 🏠 **Home**

- **`GET /`** → Returns a welcome message.

---

### 📌 **Deposition & Pinecone API**

#### 📝 **Talk to Depo**

- **`GET /depo/add/<depoiq_id>`**
  - Fetches a deposition summary from GraphQL and stores it in Pinecone.
  - **Parameters:**
    - `depoiq_id` (required): The ID of the deposition to fetch.
  - **Response:**
    - 200: Success message.
    - 500: Internal server error.

#### 🔍 **Query Summaries from Pinecone**

- **`POST /depo/talk`**
  - Query Pinecone to retrieve the most relevant deposition summaries using a user query.
  - **Request Body:**

    ```json
    {
      "depo_id": "67ab0109c9cff446cdcbc1b0",
      "user_query": "What did Jim Smith say about price increases?"
    }
    ```

  - **Response:**
    - 200: Returns the top matching summaries from Pinecone.
    - 500: Internal server error.

#### ✅ **Validate the Answer**

- **`POST /depo/answer_validator`**
  - Validates the answer to a question using deposition summaries.
  - **Request Body:**

    ```json
    {
      "questions": ["What is the impact of price increases?"],
      "depoiq_id": "67ab0109c9cff446cdcbc1b0",
      "category": "all"
    }
    ```

  - **Response:**
    - 200: Returns the validation result.
    - 500: Internal server error.

#### 🧠 **Ask Ami Agent**

- **`POST /depo/ask-ami-agent`**
  - Asks the Ami agent for information based on user queries.
  - **Request Body:**

    ```json
    {
      "depoiq_ids": ["67ab0109c9cff446cdcbc1b0"],
      "user_query": "What was discussed about price increases?"
    }
    ```

  - **Response:**
    - 200: Returns the Ami agent response.
    - 500: Internal server error.

#### 🧑‍💼 **Get Depo by ID**

- **`GET /depo/get/<depoiq_id>`**
  - Retrieves a deposition by its ID.
  - **Parameters:**
    - `depoiq_id` (required): The ID of the deposition.
    - `category` (optional): Filter results by category (e.g., `summary`, `transcript`).
  - **Response:**
    - 200: Returns the deposition details.
    - 500: Internal server error.

#### 🔑 **JWT Authentication**

#### 🔹 **Generate a Token**

- **Function**: `generate_token()`
  - Generates a **JWT token** with a payload containing `userId`, `companyId`, and `role`.
  - **Response:**
    - 200: Returns the JWT token.
    - 500: Internal server error.

---

## 🌍 Modal Integration for Serving & Deployment

This API is also deployed using **Modal**, a serverless platform for deploying APIs.

### 🔹 **Deploying with Modal**

To deploy your Flask API to Modal, follow these steps:

1. Ensure that you have a **Modal account** and the **Modal CLI** installed.
2. The Modal configuration is set up in the `main_modal.py` file, which defines the serverless app.
3. Use **Modal**'s deployment feature to deploy the app.

To deploy your app:

```bash
modal deploy main_modal.py
```

To run your app on modal:

```bash
modal serve main_modal.py
```

### 🔹 **Running the API with Modal**

Once deployed, your API will be hosted serverlessly, and you can access it through the provided URL in the Modal dashboard.

---

## 📜 License

This project is licensed under the **MIT License**.
# 🛠️ Flask API for Deposition Summaries & Pinecone Search

## 📖 Overview

This is a **Flask API** that provides endpoints to:

- **Store deposition summaries** in Pinecone
- **Query Pinecone for relevant summaries** using OpenAI embeddings
- **Manage items** (CRUD operations)
- **Authenticate using JWT**
- **Fetch deposition summaries from a GraphQL API**

The API is documented using **Flasgger**, and you can access the documentation here:  
🔗 **[API Documentation](https://ask-ami-across-platform.vercel.app/apidocs/)**

---

## 🚀 Features

✅ **GraphQL Integration**: Fetches deposition summaries from an external GraphQL API.  
✅ **Pinecone Vector Search**: Stores and retrieves deposition summaries using **AI-powered embeddings**.  
✅ **OpenAI Embeddings**: Uses **text-embedding-3-small** model to create vector representations of text.  
✅ **Flask & Flasgger**: API documentation and Swagger UI support.  
✅ **JWT Authentication**: Generates and validates access tokens.  
✅ **CRUD Operations**: Simple item management API.

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

---

## 🔑 Environment Variables

Create a `.env` file and set up your **API keys**:

```ini
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
JWT_SECRET_KEY=your-jwt-secret-key
GRAPHQL_URL=https://backend-graphql-webapp-development.up.railway.app/graphql
```

You can find a template in `.env.example`:

```ini
# Environment variable example file
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
JWT_SECRET_KEY=your-jwt-secret-key
GRAPHQL_URL=https://backend-graphql-webapp-development.up.railway.app/graphql
```

---

## ▶️ **Running the API**

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

#### 📝 **Add a Summary to Pinecone**

- **`GET /add-summaries/<depoIQ_ID>`**
- Fetches a deposition summary from GraphQL and stores it in Pinecone.

#### 🔍 **Query Summaries from Pinecone**

- **`POST /talk-summary`**
- Request Body:

```json
{
  "depo_id": "67ab0109c9cff446cdcbc1b0",
  "user_query": "What did Jim Smith say about price increases?"
}
```

- Returns the **top matching summaries** from Pinecone.

---

### 🔑 **JWT Authentication**

#### 🔹 Generate a Token

- **Function**: `generate_token()`
- Generates a **JWT token** with a payload containing `userId`, `companyId`, and `role`.

---

### 📦 **Item Management (CRUD)**

#### 📋 Get All Items

- **`GET /items`**

#### 🔎 Get Item by ID

- **`GET /items/<item_id>`**

#### ➕ Add a New Item

- **`POST /items`**
- Request Body:

```json
{
  "name": "New Item",
  "description": "This is a new item"
}
```

#### ✏️ Update an Item

- **`PUT /items/<item_id>`**
- Request Body:

```json
{
  "name": "Updated Item",
  "description": "Updated description"
}
```

#### ❌ Delete an Item

- **`DELETE /items/<item_id>`**

---

## 📝 Example API Calls

### 🔹 Store a Deposition in Pinecone

```bash
curl -X GET "http://127.0.0.1:5000/add-summary/67ab0109c9cff446cdcbc1b0"
```

### 🔹 Query a Summary

```bash
curl -X POST "http://127.0.0.1:5000/talk-summary" -H "Content-Type: application/json" -d '{
  "depo_id": "67ab0109c9cff446cdcbc1b0",
  "user_query": "What was the impact of price increases?"
}'
```

### 🔹 Get All Items

```bash
curl -X GET "http://127.0.0.1:5000/items"
```

---

## 🌍 API Documentation

🔗 **Swagger UI:** [API Docs](https://ask-ami-across-platform.vercel.app/apidocs/)

---

## 📜 License

This project is licensed under the **MIT License**.

---

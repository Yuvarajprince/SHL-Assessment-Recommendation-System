
```markdown
# SHL Assessment Recommendation System

## Overview

This repository contains an end-to-end implementation of an assessment recommendation system built using SHL’s public product catalog. The system accepts a free-text query, such as a job description or hiring requirement, and returns relevant SHL assessments using semantic retrieval techniques.

The solution covers data collection, embedding-based retrieval, intent-aware re-ranking, a REST API, and an evaluation pipeline. The implementation focuses on robustness, clarity, and strict adherence to the formats specified in the assignment documentation.

## Key Capabilities

- Scrapes SHL Individual Test Solutions from the public product catalog  
- Semantic retrieval using sentence embeddings and FAISS  
- Intent-aware re-ranking to balance assessment types  
- REST API implemented using FastAPI  
- Swagger UI for interactive API testing  
- Evaluation using Recall@10  
- CSV output generated in the exact submission format  

## Repository Structure

shl_recommender/
│
├── api/
│   └── app.py                  # FastAPI application
│
├── scraper/
│   └── scrape_shl.py            # SHL product catalog scraper
│
├── embeddings/
│   └── build_index.py           # Embedding generation and FAISS index creation
│
├── recommender/
│   ├── retrieve.py              # Semantic retrieval logic
│   └── rerank.py                # Intent-aware re-ranking logic
│
├── evaluation/
│   ├── evaluate.py              # Recall@10 evaluation
│   └── generate_submission.py   # Submission CSV generator
│
├── data/
│   └── shl_catalog.csv          # Scraped catalog data
│
├── requirements.txt
└── README.md

## Data Collection

The SHL product catalog is scraped using a controlled crawler implemented in `scraper/scrape_shl.py`.  
Only Individual Test Solutions are collected, as required by the task.

The following attributes are extracted:
- Assessment name  
- Assessment URL  
- Remote support availability  
- Adaptive support availability  
- Test type  

Description and duration fields are intentionally excluded. These fields are inconsistently available across product pages and require deeper per-page crawling, which increases blocking risk. The recommendation pipeline does not rely on these attributes.

The scraped data is stored as a structured CSV file and used for downstream processing.

## Retrieval and Recommendation Approach

Assessment metadata is converted into semantic embeddings using a pre-trained sentence transformer model. A FAISS index is built to enable efficient similarity-based retrieval.

When a query is received:
1. The query is embedded using the same model  
2. The FAISS index retrieves the most relevant assessments  
3. A re-ranking step balances results based on inferred query intent  

This approach allows the system to generalize beyond keyword matching and handle unseen queries effectively.


## API Usage

The API is implemented using FastAPI and exposes the following endpoints.

## Health Check
```
GET /health
```

Returns service status.

### Recommendation Endpoint
```

POST /recommend

````

Request body:
```json
{
  "query": "Looking for a Java developer who can collaborate with business teams"
}
````

Response:

```json
[
  {
    "name": "Java Platform Enterprise Edition 7 (Java EE 7)",
    "url": "https://www.shl.com/products/product-catalog/view/java-platform-enterprise-edition-7-java-ee-7/",
    "description": "",
    "duration": 0,
    "adaptive_support": "No",
    "remote_support": "No",
    "test_type": ["K"]
  }
]
```

Swagger UI is available at:

```
/docs
```

## Evaluation

Due to the absence of labeled relevance data, a proxy evaluation strategy is used.

Recall@10 is selected as the evaluation metric. Assessments sharing the same test type are treated as relevant to each other. While approximate, this assumption is reasonable and explicitly documented.

Evaluation logic is implemented in `evaluation/evaluate.py`.
Final predictions are exported in the exact CSV format specified in the assignment appendix.

## Running the Project Locally

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```


## Deployment

The application can be deployed on free platforms such as Render.
A public deployment exposes:

* Swagger UI as a web-based interface
* A publicly accessible recommendation API


## Notes

* The system prioritizes robustness and clarity over aggressive scraping
* All output formats strictly follow the assignment specifications
* The repository includes both implementation and evaluation logic


## Author

Developed as part of a technical assignment focused on building a retrieval-based recommendation system using SHL’s product catalog.

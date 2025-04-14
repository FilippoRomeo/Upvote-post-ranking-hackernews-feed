# 🧠 Hacker News Upvote Prediction

This project aims to predict the upvote score of posts on [Hacker News](https://news.ycombinator.com/) using their **titles only**. It uses **PyTorch**, **Word2Vec-style embeddings**, and **regression models** to learn meaningful representations of titles and predict engagement.

---

## 📦 Features

- Connects to a live PostgreSQL database of Hacker News posts
- Tokenizes titles and trains word embeddings (CBOW or Skip-gram)
- Implements a regression model to predict upvotes
- Portable development environment via Docker and JupyterLab
- Modular Python structure for easy extension

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/hn-upvote-predictor.git
cd hn-upvote-predictor

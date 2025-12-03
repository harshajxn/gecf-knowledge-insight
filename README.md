# GECF Knowledge Insight Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent full-stack web application designed to automate the extraction and summarization of key insights from large PDF documents, with a specific focus on GECF (Gas Exporting Countries Forum) member countries.


## Table of Contents

- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Key Features](#key-features)
- [How It Works: The Technical Pipeline](#how-it-works-the-technical-pipeline)
- [Tech Stack](#tech-stack)
- [Local Development Setup](#local-development-setup)
- [Deployment](#deployment)
- [Project Limitations](#project-limitations)

## The Problem

Analysts often need to review lengthy reports and publications to find specific information about GECF member countries. This is a time-consuming, manual process to find relevant sections and then summarizing the key takeaways.

## The Solution

The GECF Knowledge Insight Platform is a web-based tool that automates this entire workflow. Users can simply upload one or more PDF documents, and the application's backend pipeline intelligently parses, filters, and summarizes the content, presenting a concise and relevant analysis back to the user in seconds.

## Key Features

- **Multi-File PDF Upload:** A simple and intuitive interface for uploading multiple PDF documents at once.
- **Intelligent Content Filtering:** Automatically identifies and processes only the pages that contain relevant keywords related to GECF member countries.
- **AI-Powered Summarization:** Utilizes the high-speed Llama 3.3 70B model via the Groq API to generate expert-level summaries of the filtered content.
- **Image Extraction & Optimization:** Extracts embedded images from the PDFs and resizes them on-the-fly into web-friendly thumbnails to ensure fast performance.
- **Responsive UI:** A clean and responsive user interface that presents the results in easy-to-read cards.

## How It Works: The Technical Pipeline

The application follows a robust, multi-step process for each analysis request:

1.  **PDF Deconstruction:** The uploaded PDF is received by the Flask backend and immediately processed by the **PyMuPDF** library to extract raw text and embedded images from every page.
2.  **Performance Optimization:** A critical step for handling large files. The **Pillow** library takes the high-resolution extracted images and resizes them into lightweight JPEG thumbnails. This drastically reduces the final data payload and prevents server timeouts.
3.  **Content Filtering:** The application efficiently iterates through the extracted text a single time to identify only the pages containing GECF member country names, creating a focused `context`.
4.  **AI Summarization:** This relevant `context` is sent to the Llama 3 model using the **LangChain** framework. A carefully crafted prompt instructs the AI to act as a geopolitical energy analyst, ensuring a high-quality, concise summary.
5.  **Data Response:** The server packages the summary, a list of found countries, and the optimized Base64-encoded image thumbnails into a structured JSON object.
6.  **Dynamic Rendering:** The frontend JavaScript receives this JSON and dynamically renders the results on the page without requiring a page reload.

## Tech Stack

The project is built with a modern Python-based stack:

| Category          | Technology                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------- |
| **Backend**       | Python, Flask, Gunicorn                                                                                       |
| **AI & ML**       | LangChain (Orchestration), Groq API (LLM Inference), Llama 3.3 70B (Model)                                    |
| **Data Processing** | PyMuPDF (PDF Parsing), Pillow (Image Optimization)                                                            |
| **Frontend**      | HTML5, CSS3, Vanilla JavaScript                                                                               |
| **Deployment**    | Render (Cloud Platform), Git/GitHub (Version Control)                                                         |

## Local Development Setup

To run this project on your local machine, follow these steps:

#### Prerequisites

-   Git
-   Python 3.9+ and `pip`

#### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/harshajxn/gecf-knowledge-insight.git
    cd gecf-knowledge-insight
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    -   Create a file named `.env` in the root of the project.
    -   Add your Groq API key to this file:
        ```
        GROQ_API_KEY="your_secret_api_key_here"
        ```

5.  **Run the application:**
    ```bash
    flask run
    ```
    The application will be available at `http://127.0.0.1:5000`.

## Deployment

This application is configured for deployment on **Render**.

-   The `Procfile` tells Render how to run the application using the `gunicorn` web server.
-   The `render-build.sh` script is crucial for a successful deployment. It runs before the main build command and installs necessary system-level dependencies (`build-essential`, `swig`) that are required by the `PyMuPDF` library.

## Project Limitations

As a version 1.0 prototype, the application has the following limitations:

-   **Performance on Large Files:** While optimized, extremely large or image-heavy PDFs (e.g., 15+ pages) could still exceed the memory or timeout limits of the free hosting tier.
-   **No Concurrency:** The application processes requests synchronously. It can only handle one user's request at a time.
-   **Scanned PDFs:** The text extraction relies on the PDF having embedded text. It cannot process scanned documents where the text is part of an image (requires OCR).
-   **English Language Only:** The keyword filtering and AI prompt are designed for English-language documents.

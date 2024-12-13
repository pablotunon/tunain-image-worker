# tunain-page-worker

## Application Description

Tunain application allows users to co-write a book with an AI, generating both text and illustrations. Users provide prompts for extracts, and the AI completes paragraphs and generates illustrative images.

## Page Worker Overview

The image worker repository handles the generation of AI-created illustrations for the application using Stable Diffusion.

It listens to an SQS queue, processes messages and uses the backend REST API on completion.

## Features

- Generate images based on user prompts.
- Communicate with the backend for task coordination.

## Technology Stack

- Model: Stable Diffusion
- Environment: Python, PyTorch

## Commands

```
python3 worker.py
```

# Single Agent Pipeline

## Overview

This project demonstrates a simple rule-based Single Agent System.

The agent:
- Accepts a user query
- Detects its intent
- Routes it to the correct tool
- Returns the final response

## Pipeline

User Query
      ↓
Intent Detection
      ↓
Conditional Routing
      ├── Calculator
      ├── Keyword Extractor
      └── General Response
      ↓
Final Output

## Features

- Single Agent Architecture
- Conditional Routing
- Tool Calling
- Modular Code
- Easy to Extend

## Sample Inputs

Calculate 25*8

Output

200

---

Keywords Artificial Intelligence is transforming healthcare

Output

['Artificial', 'Intelligence', 'healthcare', 'transforming']

---

What is AI?

Output

General Response: What is AI?

## Technologies

- Python
- Rule-based Agent Pipeline

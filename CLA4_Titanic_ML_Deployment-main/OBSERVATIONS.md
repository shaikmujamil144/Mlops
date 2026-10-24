# Task 6 — Observations

## 1. How does Online Inference differ from Batch Inference?
Online inference predicts one input record at a time and returns a result for that request. Batch inference predicts many records together in one operation. Online inference is useful when an immediate answer is needed; batch inference is useful when a group of records must be processed efficiently.

## 2. What is the difference between Streaming Inference and Batch Inference?
Streaming inference processes records continuously as they arrive, one record at a time (this assignment simulates this with a one-second delay). Batch inference waits for a group of records and processes the group together. Streaming is useful for ongoing events; batch is useful for a fixed collection of data.

## 3. Why is FastAPI used for model serving?
FastAPI lets us expose the trained model through HTTP endpoints so another application can request predictions. It supports request-data validation with Pydantic and automatically provides interactive API documentation through Swagger UI at `/docs`.

## 4. Why should the same trained model be reused for all inference strategies?
Reusing the same saved pipeline makes the predictions comparable and consistent. It avoids accidentally using different model versions or preprocessing steps for online, batch, streaming, and API requests. It also avoids unnecessary retraining.

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tritonclient.http as httpclient

app = FastAPI(title="Triton Inference API")

TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "densenet_onnx"
MODEL_VERSION = "1"

client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

@app.get("/")
async def root():
    return {"message": "Triton Inference API is running"}

@app.post("/inference/")
async def inference(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Không thể đọc ảnh từ file upload.")

        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        print(f"Input image shape: {image.shape}")

        inputs = httpclient.InferInput("data_0", image.shape, "FP32")
        inputs.set_data_from_numpy(image)

        outputs = httpclient.InferRequestedOutput("fc6_1")

        response = client.infer(MODEL_NAME, model_version=MODEL_VERSION, inputs=[inputs], outputs=[outputs])
        result = np.squeeze(response.as_numpy("fc6_1")).flatten() 

        print("Top 5 inference outputs:", result[:5])

        probs = np.exp(result) / np.sum(np.exp(result))
        top_5_indices = np.argsort(probs)[-5:][::-1]
        top_5_confidences = probs[top_5_indices]

        results = [
            {"class": int(idx), "confidence": round(float(conf), 4)}
            for idx, conf in zip(top_5_indices, top_5_confidences)
        ]

        return JSONResponse(content={"inference": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

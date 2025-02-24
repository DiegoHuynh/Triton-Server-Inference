from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from trism import TritonModel

# Tạo ứng dụng FastAPI
app = FastAPI(title="Triton Inference API with Trism")

# Khởi tạo mô hình Triton thông qua Trism
model = TritonModel(
    model="densenet_onnx",  # Tên mô hình
    version=1,               # Phiên bản mô hình
    url="localhost:8001",    # URL của Triton Server (gRPC)
    grpc=True                # Sử dụng gRPC
)

# Kiểm tra thông tin đầu vào/đầu ra của mô hình
for inp in model.inputs:
    print(f"Input - name: {inp.name}, shape: {inp.shape}, datatype: {inp.dtype}")
for out in model.outputs:
    print(f"Output - name: {out.name}, shape: {out.shape}, datatype: {out.dtype}")

# Kiểm tra kết nối
@app.get("/")
async def root():
    return {"message": "Triton Inference API is running with Trism"}

# API Inference
@app.post("/inference/")
async def inference(file: UploadFile = File(...)):
    try:
        # Đọc file ảnh tải lên
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Không thể đọc ảnh từ file upload.")

        # Tiền xử lý ảnh
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        transformed_img = np.expand_dims(image, axis=0).astype(np.float32)

        # Inference với Triton
        outputs = model.run(data=[transformed_img])
        result = np.squeeze(outputs['fc6_1']).flatten()  # Đưa output về 1D

        # Tính xác suất và lấy top-5 kết quả
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

# Chạy ứng dụng FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

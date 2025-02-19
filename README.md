# Triton Inference Server Deployment Report

## Project Setup
### Download Triton Server
```
    docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v /home/duc_huynh/Triton-Server-Inference/model_repository:/models \
    nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models
```

### Prepare Model Repository
Create a directory `models/` and structure it as follows:
```
model_repository
|
+-- densenet_onnx
    |
    +-- config.pbtxt
    +-- 1
        |
        +-- model.onnx
```

### Write Model Configuration (`config.pbtxt`)
```txt
name: "densenet_onnx"
platform: "onnxruntime_onnx"
max_batch_size : 0
input [
  {
    name: "data_0"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 224, 224 ]
    reshape { shape: [ 1, 3, 224, 224 ] }
  }
]
output [
  {
    name: "fc6_1"
    data_type: TYPE_FP32
    dims: [ 1, 1000, 1, 1 ] 
  }
]
```

---

## Running Triton Server
```
I0219 13:39:47.443768 1 server.cc:653]
+---------------+---------+--------+
| Model         | Version | Status |
+---------------+---------+--------+
| densenet_onnx | 1       | READY  |
+---------------+---------+--------+

I0219 13:39:47.444440 1 metrics.cc:701] Collecting CPU metrics
I0219 13:39:47.444989 1 tritonserver.cc:2387]
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Option                           | Value

     |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| server_id                        | triton

     |
| server_version                   | 2.33.0

     |
| server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shared_memory cuda_shared_memory binary_tensor_data parameters statistics trace logging |
| model_repository_path[0]         | /models

     |
| model_control_mode               | MODE_NONE

     |
| strict_model_config              | 0

     |
| rate_limit                       | OFF

     |
| pinned_memory_pool_byte_size     | 268435456

     |
| min_supported_compute_capability | 6.0

     |
| strict_readiness                 | 1

     |
| exit_timeout                     | 30

     |
| cache_enabled                    | 0

     |
+----------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

I0219 13:39:47.453632 1 grpc_server.cc:2450] Started GRPCInferenceService at 0.0.0.0:8001
I0219 13:39:47.453879 1 http_server.cc:3555] Started HTTPService at 0.0.0.0:8000
I0219 13:39:47.500264 1 http_server.cc:185] Started Metrics Service at 0.0.0.0:8002

```

## Inference Result
```
['11.481906:14' '11.101565:92' '8.271426:88' '7.760969:17' '7.278370:90']
```

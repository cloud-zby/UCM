# Setup
## 安装
```bash
sudo docker pull vllm/vllm-openai:v0.9.2
sudo mkdir /home/zby/models
sudo mkdir /home/zby/storage
sudo docker run --gpus all --network=host --ipc=host -v /home/zby/models:/home/model -v /home/zby/storage:/home/storage --entrypoint /bin/bash --name vllm-openai -it vllm/vllm-openai:v0.9.2
git clone --depth 1 --branch main https://github.com/ModelEngine-Group/unified-cache-management.git
cd unified-cache-management
export PLATFORM=cuda
pip install -v -e . --no-build-isolation
cd $(pip show vllm | grep Location | awk '{print $2}')
git apply /vllm-workspace/unified-cache-management/ucm/integration/vllm/patch/0.9.2/vllm-adapt.patch
git apply /vllm-workspace/unified-cache-management/ucm/integration/vllm/patch/0.9.2/vllm-adapt-sparse.patch # ERROR
```
## 启动
```bash
sudo docker run --gpus all --network=host --ipc=host -v /home/zby/models:/home/model -v /home/zby/storage:/home/storage --entrypoint /bin/bash --name vllm-openai -it vllm/vllm-openai:v0.9.2
```
## 进入容器
```bash
sudo docker exec -it vllm-openai /bin/bash
```
## DEMOs
### Prefix Cache
#### DRAM Store
##### Offline Inference
CUDA_VISIBLE_DEVICES根据实时使用情况调整
```bash
cd /vllm-workspace/unified-cache-management/examples/
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1
python3 offline_inference.py
```
##### Online Inference
```bash
cd /vllm-workspace/unified-cache-management/examples/
export PYTHONHASHSEED=123456
export CUDA_VISIBLE_DEVICES=2,3
ray start --head --num-cpus=8 --num-gpus=2 --port=6380
vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 20000  --tensor-parallel-size 2  --gpu_memory_utilization 0.4  --trust-remote-code  --port 7800  --kv-transfer-config  '{
    "kv_connector": "UnifiedCacheConnectorV1",
    "kv_connector_module_path": "ucm.integration.vllm.uc_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "ucm_connector_name": "UcmDramStore",
        "ucm_connector_config": {
            "max_cache_size": 1073741824,
            "kv_block_size": 65536
        }
    }
}'
```
当出现
```bash
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```
之后，就可以进行使用
```bash
curl http://localhost:7800/v1/completions  -H "Content-Type: application/json" -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "Shanghai is a what kind of city?",
        "max_tokens": 100,
        "temperature": 0
    }'
```
#### PD Disaggregation
##### 1p1d
运行prefill
```bash
export PYTHONHASHSEED=123456
export HF_ENDPOINT=https://hf-mirror.com
clear
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 20000 --tensor-parallel-size 1 --gpu_memory_utilization 0.4 --trust-remote-code --enforce-eager --no-enable-prefix-caching --port 7800 --block-size 128 --kv-transfer-config '{
    "kv_connector": "UnifiedCacheConnectorV1",
    "kv_connector_module_path": "ucm.integration.vllm.uc_connector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "ucm_connector_name": "UcmNfsStore",
        "ucm_connector_config": {
            "storage_backends": "/vllm-workspace/unified-cache-management/data",
            "transferStreamNumber":32
        }
    }
}'
```
运行decode
```bash
export PYTHONHASHSEED=123456
export HF_ENDPOINT=https://hf-mirror.com
clear
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 20000 --tensor-parallel-size 1 --gpu_memory_utilization 0.4 --trust-remote-code --enforce-eager --no-enable-prefix-caching --port 7801 --block-size 128 --kv-transfer-config '{
    "kv_connector": "UnifiedCacheConnectorV1",
    "kv_connector_module_path": "ucm.integration.vllm.uc_connector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
        "ucm_connector_name": "UcmNfsStore",
        "ucm_connector_config": {
            "storage_backends": "/vllm-workspace/unified-cache-management/data",
            "transferStreamNumber":32
        }
    }
}'
```
启动代理服务器
10.90.1.28是hostname -I之后查询得到的本机ip地址
```bash
cd /vllm-workspace/unified-cache-management/ucm/pd/
python3 toy_proxy_server.py --host localhost --port 7802 --prefiller-host 10.90.1.28 --prefiller-port 7800 --decoder-host 10.90.1.28 --decoder-port 7801
```
简单测试
```bash
curl http://localhost:7802/v1/completions -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "What date is today?",
    "max_tokens": 2000,
    "temperature": 0
}'
```
Benchmark测试
```bash
cd /vllm-workspace/benchmarks
export HF_ENDPOINT=https://hf-mirror.com
clear
python3 benchmark_serving.py --backend vllm --dataset-name random --random-input-len 4096 --random-output-len 1000 --num-prompts 10 --ignore-eos --model Qwen/Qwen2.5-1.5B-Instruct --tokenizer Qwen/Qwen2.5-1.5B-Instruct --host localhost --port 7802 --endpoint /v1/completions --request-rate 1
```
---

# Unified Cache Management
Docs: [https://hackmd.io/@peter-john/B1E9j-PCle](https://hackmd.io/@peter-john/B1E9j-PCle)






























## Reference
UCM code link: [https://github.com/ModelEngine-Group/unified-cache-management.git](https://github.com/ModelEngine-Group/unified-cache-management.git)
UCM documentation link: [https://unified-cache-management.readthedocs.io/en/latest/](https://ucm.readthedocs.io/en/latest/index.html)




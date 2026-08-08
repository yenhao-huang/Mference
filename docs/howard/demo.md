
## Generate
```bash
cd /Users/yenhaohuang/Desktop/side_project/Mference

**Server**
.build/release/MferenceServer \
--model scratch/deepseekv4flash.gturbo \
--port 3001


**CLI**
.build/release/MferenceCLI \
--model scratch/deepseekv4flash.gturbo \
--chat \
--max-context 4096 \
--expert-cache-slots 16 \
--rdadvise adaptive \
--verify trusted-receipt
```

## Build
download model and convert to .gturbo
```
swift run -c release MferenceRepack \
  --model qwen36 \
  --output scratch/qwen36.gturbo
```

## 相關檔案
部署完成，DeepSeek-V4-Flash 284B 已在這台 Apple M4／32 GiB Mac 上成功運行。
- 模型：DeepSeek-V4-Flash 284B-A13B 2-bit DQ 
- 路徑：Mference/scratch/deepseekv4flash.gturbo
- 完整驗證：50 files、96,689,373,952 bytes，全數通過
- 實機推論：24 tokens，decode 9.61 秒，2.497 toks                                                                                                                                                   
- Prompt：The capital of Taiwn is                                                                                                                                                  
- 輸出開頭：Taipei. The official language is Mandarin...                                                                                                                                         
- 磁碟配置：約 90 GiB，剩餘 161 GiB                                                                                                                                                - 全流程耗時：約 13 小時 1 分鐘

交付文件：
- Mference/deployment/deepseek-v4-flash/README.md
- Mference/deployment/deepseek-v4-flash/TROUBLESHOOTING.md
- Mference/deployment/deepseek-v4-flash/finalize.log
- Mference/scratch/deepseekv4flash.gturbo/verified-install.json

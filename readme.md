# 🚀 **MH-CDNet: Map- and History-Aided Change Detection of Traffc Signs in High-Defnition Maps**  

## 📌 **项目概述**  
本项目提出了一种 **交通标志变化检测系统**，涵盖 **数据处理、模型训练** 及 **模型微调** 三大模块。该系统利用 **多传感器数据融合**，结合 **历史轨迹与高精度地图信息**，实现对交通标志变化的智能检测与推理，并优化模型性能以适应实际应用场景。  

---

## 📂 **数据处理**

### 🛠 **数据解析**
1️⃣ **批量数据解析**  
   本模块用于 **自动解析** 采集到的 **位置数据与图像数据**，并将解析结果存储至指定路径：
   ```bash
   python data_parser/parser/auto_main.py
   ```
   - **生成日志文件**：📜 `process_files_{i}.log`
   - **输出路径**：📂 `<server_path>/parser/insdata`

2️⃣ **摄像头参数解析**  
   本模块负责摄像头参数的提取与处理，以确保数据的几何校正与准确性：
   - 使用 **CAVision** 软件进行摄像头参数解析  
   - 提取与对应摄像头匹配的标定文件
   （⚠ **操作需在虚拟机环境下完成**）

3️⃣ **数据分类**  
   解析完成后，需按照 **车辆类型、时间及行驶路线** 对 `dat` 文件进行分类，以便后续处理：
   ```bash
   python data_parser/process_scene/shutil_dat.py
   ```
   - 分类后的数据存储至：📂 `<server_path>/extracted_data`

4️⃣ **场景筛选**  
   - 将 **定位数据投影至 QGIS**，并 **人工筛选** 可能包含交通标志的 **关键路口场景**。  
   - 依据筛选出的 **场景坐标范围** 进行数据过滤，提取符合条件的 **重访数据**：  
   ```bash
   python data_parser/process_scene/shutil_file_to_scene.py
   ```  
   - **场景定义文件**：📜 `<server_path>/scenes/scenes.csv`  
     文件示例：  
     `1.0 xxx xxx xxx xxx`（经纬度范围）  
   - **输出路径**：📂 `<server_path>/scenes`  
     文件夹结构如下：  
     ```
     <server_path>/scenes  
         ├── 1.0  
         │   ├── xxx.dat  
         │   └── xxx.dat  
         ├── 2.0  
         └── ...  
         └── scenes.csv  
     ```  

5️⃣ **数据插值**  
   由于原始图像记录频率与位置信息记录频率不同，为确保数据一致性，需进行 **插值处理**：
   ```bash
   python data_parser/process_scene/interplate.py
   ```
   - **生成插值文件**：📜 `<dat_path>/interpolated_poses.json`

6️⃣ **目标检测**  
   本模块采用 **YOLO 目标检测模型** 识别场景中的交通标志，并存储检测结果：
   ```bash
   sh data_parser/process_scene/transfer_data.sh
   ```
   - **检测结果文件**：📜 `<dat_path>/detections.json`
   - **输出路径**：📂 `<server_path>/detected_scenes`

7️⃣ **坐标计算**  
   计算 **场景中同类别交通标志** 的坐标，并视为同一实例：
   ```bash
   python data_parser/process_scene/cal_coordinates.py
   ```
   - **计算结果文件**：📜 `<dat_path>/coordinates.json`

     最终得到dat文件夹结构如下：  
     ```
      <dat_path>/
         ├── CD701_000013_2024-05-01_11-30-47/
         ├── imu_data.json
         ├── interpolated_poses.json
         ├── pos_imu_data.json
         ├── sda-encode_srv-EncodeSrv.EncodeH265-CAM9/
               ├── 6633406207.jpg
               ├── 6636839542.jpg
               ├── 6640306210.jpg
               ├── 6643739544.jpg
               ├── 6647206212.jpg
               ├── 6650639546.jpg
               └── ...
         ├── sda-encode_srv-EncodeSrv.EncodeH265-CAM9_stream.bin
         ├── sda-ins_parser-localization.InsData.dat
         ├── sda-ins_parser-localization.InsData-index.dat
         ├── wheel_speed_data.json
         ├── coordinates.json
         └── detections.json
     ``` 
     
8️⃣ **数据集构建**  
   依据 **场景坐标文件** 构建标准化的 **交通标志数据集**：
   ```bash
   python data_parser/process_sign/sta_coordinates.py
   ```
   - **筛选与重命名** 每个路牌类别的文件夹：
     ```bash
     python data_parser/process_sign/filter_traffic_sign_dataset.py
     ```
   - **统计交通标志数据集信息**（包括类别分布、标志数量、重访次数及误差）：
     ```bash
     python data_parser/process_sign/sta_traffic_sign_dataset.py
     ```
   - **输出路径**：📂 `<server_path>/traffic_sign_dataset`
     文件夹结构如下：  
     ```
     <server_path>/traffic_sign_dataset/
      ├── class_2_i5/
         ├── sign_1/
            ├── errors.json
            ├── scenes.json
            ├── visit_5_CD701_000013_2024-05-01_14-10-12/
                  ├── 6633406207.jpg
                  ├── 6636839542.jpg
                  ├── 6640306210.jpg
                  ├── 6643739544.jpg
                  ├── 6647206212.jpg
                  ├── 6650639546.jpg
                  ├── ...
                  ├── data.json
                  └── max_confidence_image.jpg (作为此次重访的代表)
            └── ... (不同重访实例)
         └── ... (不同路牌实例)
      └── ... (不同路牌类别)
     ```
     data.json示例如下：
     ```
     {
      "scene_folder": "./detected_scenes/26.0/CD701_000013_2024-05-01_14-10-12",
      "class": 2,
      "timestamp": 2837.274168,
      "confidence": 0.9496677673,
      "WGS_Lon": 106.5609265943,
      "WGS_Lat": 29.5879404183,
      "WGS_Alt": 849.1702551395,
      "UTM_X": 651175.4710579601,
      "UTM_Y": 3274144.176355629,
      "UTM_Z": 849.1702551395,
      "max_confidence_image_path": "./detected_scenes/26.0/CD701_000013_2024-05-01_14-10-12/sda-encode_srv-EncodeSrv.EncodeH265-CAM9/2837274168.jpg",
      "x1": 2330.0515136719,
      "y1": 956.8365478516,
      "x2": 2432.85546875,
      "y2": 1061.2010498047,
      "num of imgs": 369,
      "list of imgs": [
         "2820874163.jpg",
         "2820907497.jpg",
         "2820974163.jpg",
         ...
         ]
      }
     ```
---

### 🎛 **数据增强**
1️⃣ **随机数据增强**  
   本模块根据预设的 **事件类型数量**，对数据集进行 **随机增强**：
   ```bash
   python data_parser/augment/random_add.py
   ```

2️⃣ **数据筛选与统计**  
   - **人工筛选** 增强后的数据，找到不同类型事件：
     ```bash
     python data_parser/augment/find_FP.py
     python data_parser/augment/find_NP.py
     ```
   - **统计最终增强后的数据集**：
     ```bash
     python data_parser/augment/lookup.py
     ```

---

## 🔥 **模型设计**

### 📊 **数据集构建**
- **数据格式**  
  每条记录包含 **车辆对交通标志的一次观测**，并结合 **高精度地图** 与 **历史数据** 作为模型输入。  
- **数据集划分**  
  - 训练集: 测试集: 验证集 = **7:2:1**  
  - 采用 **随机打乱策略**，确保模型泛化能力  

📌 **数据字段说明**  
| 列名 | 描述 |
|----------------|------------------------------------------|
| **class_id** | 交通标志类别唯一标识符（如 `"i10"`） |
| **sign_id** | 具体交通标志的唯一标识符 |
| **visit_id** | 记录车辆对该标志的某次观测 |
| **coordinates** | GPS 坐标（`[longitude, latitude]` 格式） |
| **image_path** | 观测图像文件路径 |
| **bbox** | 目标在图像中的边界框 `[x_min, y_min, x_max, y_max]` |
| **timestamp** | 观测时间 (`YYYY-MM-DD HH:MM:SS`) |
| **confidence** | 目标检测置信度（0~1） |
| **image_quality_index** | 图像质量评分 |
| **Weather** | 拍摄时的天气情况（0 = 晴天，1 = 雨天等） |
| **Dayparts** | 拍摄时段（0 = 上午，1 = 下午等） |
| **Blur**             | 图像模糊程度（0 = 无模糊，1 = 有模糊） |  
| **Occlusion**        | 目标遮挡情况（0 = 无遮挡，1 = 有遮挡） |
| **visit_num_id** | `visit_id` 对应的数值编号 |
| **class_num_id** | `class_id` 对应的数值编号 |
| **event_type** | 观测事件类型（如 `"tp"`） |
| **existence** | 交通标志是否存在（1 = 存在，0 = 缺失） |

---

### 🎯 **模型训练**
1️⃣ **完整模型训练**（基于 **历史轨迹 + 地图信息**）：
   ```bash
   python model/train_all.py
   ```

2️⃣ **仅基于历史轨迹训练**：
   ```bash
   python model/train_re.py
   ```

3️⃣ **仅基于地图训练**：
   ```bash
   python model/train_map.py
   ```

4️⃣ **模型测试**：
   ```bash
   python model/test.py
   ```

---

### 🔬 **对比实验**
1️⃣ **贝叶斯方法**：
   ```bash
   python model/bayes.py
   ```

2️⃣ **Dempster-Shafer 理论**：
   ```bash
   python model/ds.py
   ```

---

## ⚙ **环境配置**
📦 **依赖安装**
- 方式一：使用 YAML 文件创建环境（适用于联网环境）
```bash
conda env create -f environment.yml
```

- 方式二：使用 Conda Pack 打包环境（适用于离线部署）   
  已使用 conda-pack 打包运行环境为 **`MH-CDNet-env.tar.gz`**，适用于无网络环境下快速部署。
   - 📦 **环境包路径**: `ssh://125.220.153.57:/home/gyx/project/MH-CDNet/MH-CDNet-env.tar.gz`


🔧 **变量定义**
- 📂 **`<dat_path>`**：指每个 `.dat` 文件解析后生成的对应文件夹路径。  
- 🖥 **`<server_path>`**：指原始数据服务器存储路径 `ssh://125.220.153.26/mnt/storage/gyx/`。

🌍 **服务器路径**
- **数据集代码**：`ssh://125.220.153.57:/home/gyx/project/MH-CDNet/`
- **完整代码**：`ssh://125.220.153.57:/home/gyx/project/MH-CDNet_all/`
- **打包数据集**（约 **200GB**）：`ssh://125.220.153.57:/home/gyx/project/MH-CDNet.tar.gz`  

⚠ **由于文件大小限制，GitHub 目前仅提供整理后的数据表格与核心代码。运行代码还需要以下资源：clip_model、traffic_sign_dataset、traffic_sign_dataset_sample 和 traffic_sign_dataset_mask。下载请参考数据集代码路径**

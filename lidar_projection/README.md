# Directory Tree
```
lidar_projection
├─ CMakeLists.txt
├─ config
│  ├─ config.rviz
│  └─ parameters.yaml
├─ frames.pdf
├─ launch
│  └─ projection.launch
├─ package.xml
├─ requirements.txt
└─ src
   ├─ PointFlow
   │  ├─ dataLoader.py
   │  ├─ main.py
   │  ├─ models
   │  │  ├─ FlowNet2S.py
   │  │  ├─ OrientaionNet.py
   │  │  ├─ PointFlow.py
   │  │  ├─ TranslationNet.py
   │  │  └─ __init__.py
   │  └─ submodules.py
   ├─ __init__.py
   ├─ dataset
   │  ├─ __init__.py
   │  ├─ custom_sequence
   │  │  ├─ seq00
   │  │  │  ├─ img
   │  │  │  └─ pose_seq00.txt
   │  │  ├─ seq01
   │  │  │  ├─ img
   │  │  │  └─ pose_seq01.txt
   │  │  ├─ seq02
   │  │  │  ├─ img
   │  │  │  └─ pose_seq02.txt
   │  │  ├─ seq04
   │  │  │  ├─ img
   │  │  │  └─ pose_seq04.txt
   │  │  ├─ seq05
   │  │  │  ├─ img
   │  │  │  └─ pose_seq05.txt
   │  │  ├─ seq06
   │  │  │  ├─ img
   │  │  │  └─ pose_seq06.txt    
   │  │  ├─ seq07
   │  │  │  ├─ img
   │  │  │  └─ pose_seq07.txt     
   │  │  ├─ seq08
   │  │  │  ├─ img
   │  │  │  └─ pose_seq08.txt    
   │  │  ├─ seq00
   │  │  │  ├─ img
   │  │  │  └─ pose_seq09.txt    
   │  │  ├─ seq10
   │  │  │  ├─ img
   │  │  │  └─ pose_seq10.txt          
   │  ├─ ground_truth
   │  │  ├─ 00.txt
   │  │  ├─ 01.txt
   │  │  ├─ 02.txt
   │  │  ├─ 03.txt
   │  │  ├─ 04.txt
   │  │  ├─ 05.txt
   │  │  ├─ 06.txt
   │  │  ├─ 07.txt
   │  │  ├─ 08.txt
   │  │  ├─ 09.txt
   │  │  ├─ 10.txt
   │  │  ├─ relative_00.txt
   │  │  ├─ relative_01.txt
   │  │  ├─ relative_02.txt
   │  │  ├─ relative_04.txt
   │  │  ├─ relative_05.txt
   │  │  ├─ relative_06.txt
   │  │  ├─ relative_07.txt
   │  │  ├─ relative_08.txt
   │  │  ├─ relative_09.txt
   │  │  └─ relative_10.txt
   │  ├─ plot_odom.py
   │  ├─ test.py
   │  └─ tools.py
   ├─ spherical_projection.py
   └─ tools.py
```

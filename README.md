# Autonomous Driving Student
- Code for Autonomous Driving(M3621.000300) class
- 별표(*) 표시된 폴더에 학생들이 채워야 할 코드가 있음.
```bash
 ros2_ws
    ├── build
    ├── install
    ├── log
    ├── src
        ├── f1tenth
        │    ├── * algorithm
        │    ├── configs
        │    ├── installation
        │    ├── leaderboard.py
        │    ├── main_il.py
        │    ├── main_rl.py
        │    ├── README.md
        │    ├── requirements.txt
        │    ├── * utils
        │    └── visualize.py
        └── f1tenth_ros
             ├── ackermann_mux
             ├── * f1tenth_stack
             ├── LICENSE
             ├── README.md
             ├── scripts
             ├── teleop_tools
             └── vesc

```

## Installation
1. ```cd ros2_ws```
2. 위와 같은 구조로 폴더 형성 **( 다운받은 Repository를 src 로 변경!!! )**
    - Ros2 구조이기 때문
3. ```colcon build --symlink-install```
    - 코드가 채워져 있지 않아도 빌드에는 문제 없음.
4. ```cd src/f1tenth/installation && pip install -e .```

# PickOrange运行入口：智能体指令

本目录负责 PickOrange 的策略训练、模型服务、仿真评测及演示采集。开发约定见[根目录指南](../../AGENTS.md)，运行入口见 [PickOrange运行指南](README.md)，环境和资产见[环境指南](../../docs/PICK_ORANGE.md)与[资源清单](../../docs/RESOURCES.md)。

- `common.py`校验协议、NPZ输入白名单和有界等待；`service.py`默认GR00T N1.5，`service_n16.py`和`service_pi05.py`各绑定自己的权重与后端；`evaluate.py`负责原生动作转换、完整回合及reset前捕获；`audit.py`独立读盘。不要将N1.6权重交给N1.5加载器，或复用π₀.₅域基座的处理器。
- 各服务在构造模型前验证实际后端版本及受管工作树；不要仅凭协议版本声明放行。收尾须先关闭SwanLab writer再关闭环境／SimulationApp，保留错误但仍尝试释放资源；补测关闭顺序和失败分支，不伪造历史SDK状态。
- `service_smolvla.py`复用同一IPC／评测／审核，固定Edge导出与LeRobot v0.4.2；完整front+wrist、原pre/post统计，不复用LiftCube缺槽配置。每次预测清缓存但不重seed，30 Hz动作只由评测器保持两tick。新增原生辅助协议只有CPU证据，不继承旧固定执行器成绩。
- Evo-1 使用项目提供的 LeRobot 后端、配套训练统计和独立实验协议。相机顺序、六维绝对 motor 目标、整块 50 步执行与 30 Hz 时钟按协议校验，共用评测和轨迹审核。
- 视觉控制脚本与特权state控制脚本从[统一指南](../../docs/SCRIPTED_POLICIES.md)定位实际源码。上游控制脚本留在冻结第三方目录，勿复制成自研视觉控制脚本；PickOrange视觉控制脚本入口为`visual/evaluate.py`，使用独立策略和评分边界；`--input-contract rgb-q-calibration`仅允许RGB、q6及已知相机标定，不读取重力补偿力、固定桌面高度或物体尺寸；运行与独立 55 秒协议见 [视觉控制脚本指南](visual/README.md)。reset 后在首次取图前刷新渲染，常规评测只保存数值。
- N1.6手臂相对量由原生处理器转为绝对motor，夹爪已为绝对量；不再加当前q。π₀.₅按checkpoint原统计解码，30 Hz动作保持两个60 Hz物理步。仿真和模型须使用同一协议，禁止为方便比较改变模型的预处理或执行节奏。
- `openvla_oft.py`是待受训包的接口，不继承LiftCube单front入口；无匹配同源双RGB、六关节动作头/投影与统计时拒绝加载。不能把新初始化组件、通用OpenVLA或LIBERO结果当作PickOrange策略能力。
- 默认协议 `protocols/groot-native-assisted.json` 允许上游夹爪控制使用物体真值，网络输入仅含协议声明的观测。该协议的成绩单列为特权辅助评测，与无特权视觉基线及课程正式评分分开记录。更改物理、辅助、相机、频率、限幅或时限时，采用新的协议名并重新评测。
- `--check`与`--help`不可创建AppLauncher或加载大模型。服务须显式`--run-service`，仿真须`--run-simulation`。文件IPC仅在受信任本地共享目录，不使用pickle或开放网络端口。
- 结果目录唯一，服务创建/evaluator独占claim；不复用旧ready、删除claim、覆盖失败或自动重跑。当前无自动resume；保留partial，不重复计入已完成seed。RNG存档不能冒称任意中途物理恢复。
- `evaluate.py`复用LiftCube包的SwanLab数值记录层，仅evaluator写run，service不重复写。查询耗时按query、结果按完整episode计数；回合记录仍须独立审核。默认offline，私有online绑定留在使用者本地；不上传RGB/HDF5/RNG、完整协议/路径或终端信息。`--check`不得导入或初始化SDK，日志修改补测`tests/test_swanlab.py`。
- 原始模型motor、转换后rad、action manager raw和实测q分开，明确scale/offset与关节顺序；本原生辅助协议不软件限幅，其他包装的动作保护不得因此放宽。
- 官方特权state控制脚本代码已在`third_party/leisaac`。其重力/阻尼/直接写状态与成功覆盖必须在采集说明中披露；不得把它的生成成功当成固定物理BC准入。
- 在各方法的运行指南中列出依赖、权重来源和命令。区分推理服务、离线行为克隆和在线残差训练的入口。
- 历史评测记录保留当时的模型、协议与场景范围，见[检验摘要](../../docs/PICKORANGE_VALIDATION.md)。

ACT 和 GR00T 的在线训练分别见 [ACT 残差指南](sac/README.md)与 [GR00T 残差指南](groot_sac/README.md)。更改训练或评测设置时同步核对训练时限、控制模式、夹爪力矩、回放及超时处理。

行为克隆从 [ACT 指南](act/README.md)进入。训练入口为 `python -m course.pickorange.act.train`。更改观测、动作、分段或时间聚合时，同时核对已发布检查点的配置，运行 `tests/test_pickorange_act_data.py`，并用相应控制模式检查导出的权重可重新加载。

修改后在根目录的独立CPU环境运行：

```bash
python course/pickorange/evaluate.py --check
python -B -m unittest discover -s tests -p test_pickorange.py -v
python -B -m unittest discover -s tests -p test_pickorange_visual.py -v
python -B -m unittest discover -s tests -p 'test_vla*.py' -v
```

`test_pickorange.py` 需 NumPy，HDF5 审核测试另需 h5py。`test_pickorange_visual.py` 的完整 CPU 检查需 NumPy、Torch、SciPy 和 OpenCV。测试使用替身模型与合成数据，实际抓放由仿真评测验收。同步 `manifests/course_files.json`、任务状态、对应 README 及影响到的根分流。

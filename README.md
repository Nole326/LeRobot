# LeRobot integration workspace

整合LeRobot、LeIsaac、IsaacLab固定版本完整源码与独立验证工具。本仓库不是Hugging Face官方LeRobot，也不是任何课程的正式环境或评分器。

公开内容仅为源码和通用说明。课程要求、报告、服务器资料、验证记录、截图、原PDF和私有配置留在本地，不进入公开历史。

## 目录

| 路径 | 内容 |
| --- | --- |
| `third_party/lerobot/` | LeRobot v0.4.2完整源码，含ACT、SmolVLA、SAC/HIL-SERL、数据/硬件工具、测试、文档 |
| `third_party/leisaac/` | LeIsaac 0.4.0固定commit完整源码 |
| `third_party/isaaclab/` | IsaacLab v2.3.0完整源码 |
| `third_party/leisaac/dependencies/IsaacLab/` | 同版本IsaacLab展开副本，避免原嵌套子模块为空 |
| `course/scripts/` | 16项独立验证工具源码，不含课程文件或实验输出 |
| `course/constraints-*.txt` | 依赖约束示例，不是通用一键安装配方 |
| `scripts/` | 完整性审计和官方测试素材恢复工具 |
| `manifests/` | 项目版本、逐文件哈希、测试素材与外部资产清单 |
| `docs/` | 通用依赖、导入与部署说明 |

全部45项上游LFS测试素材已下载核验并存入普通Git，无需额外LFS配额或递归初始化子模块。

## 获取与检查

```bash
git clone https://github.com/Nole326/LeRobot.git
cd LeRobot
python scripts/audit_repository.py
```

审计不导入仿真库、不访问GPU。进一步阅读：

- [完整依赖范围](docs/DEPENDENCIES.md)
- [部署布局与边界](docs/REPRODUCIBILITY.md)
- [导入清单](docs/IMPORT_MANIFEST.md)
- [上游许可与版权](THIRD_PARTY_NOTICES.md)

Isaac Sim镜像、独立场景资产、完整数据集与预训练模型通过官方渠道另行获取。源码齐全不等于正式动作对齐、端到端RL或完整训练恢复已经通过；现有工具仅供独立验证。请遵守所在机构的容器、GPU预约、数据和硬件安全规则，不修改宿主驱动或其他健康任务。

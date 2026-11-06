# Golden 验证报告模板

读取时机：完成参考结果设计或执行后，用于整理 PR 描述、review 回复或缺陷定位记录。

复制下方模板并删除不适用项。未执行的检查必须写 `NOT_RUN`，不能留空或记为通过。

## 模板

````markdown
## Golden Reference Validation

### Scope

- Operator:
- Interface/call path:
- Commit or diff:
- Validation objective:

### Contract

- Contract sources:
- Inputs:
- Outputs:
- Attributes:
- Known contract conflicts:

### Golden Source

- Reference API or implementation:
- Version:
- Independence rationale:
- Known semantic differences:

### Cases

| case_id | dtype/shape | attributes | data pattern | seed | comparison | status |
|---|---|---|---|---|---|---|
| G-001 |  |  |  |  |  | NOT_RUN |

### Comparison Rules

| output | method | atol | rtol | invariant | rationale |
|---|---|---:|---:|---|---|
|  | exact / tolerance / invariant | N/A | N/A |  |  |

### Commands Executed

```text
<exact commands>
````

### Results

- Reference-side result:
- NPU result:
- Maximum absolute error:
- Maximum relative error:
- Invariant checks:

### First Failing Case

- case_id:
- Input summary:
- Expected:
- Actual:
- First/max mismatch:
- Raw log:
- Suspected stage:

### Validation Status

- PASS:
- FAIL:
- BLOCKED:
- NOT_RUN:

### Unverified Items

- NPU-only checks:
- Unsupported or unknown inputs:
- Missing environment/data:
- Reviewer confirmation needed:

## 填写规则

- 命令使用仓库相对路径，不写个人目录。
- 大输入只保留生成方法、seed、摘要和失败切片，不把整块数据贴入 PR。
- 原始日志可能包含路径、设备标识或数据内容时，先按仓库安全要求脱敏。
- `PASS` 只能用于实际执行的 case。
- 修改实现后保留修复前的首个失败证据，并新增修复后结果。

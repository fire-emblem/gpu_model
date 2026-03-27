# Doc

Command executed by this usage bundle:

```bash
./build/tests/gpu_model_tests \
  --gtest_filter='RequestedShapes/*:RequestedThreadScales/*'
```

Expected artifact:

- `results/stdout.txt`

Expected behavior:

- covers the requested functional and cycle scale matrices,
- includes the large `G1024_T1024` functional cases,
- includes the requested cycle scale coverage with `blockDim.x=65` and `gridDim.x=1,64,104,1024`.

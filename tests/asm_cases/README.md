# Assembly Cases

This directory stores AMDGPU assembly fixtures used by end-to-end loader tests.

Guidelines:
- group fixtures by consumer area, for example `loader/`
- keep each file focused on one coverage theme or one larger mixed-usage topic
- prefer real assembler syntax that can be assembled by `llvm-mc`
- when adding a new fixture, prefer extending test discovery instead of embedding long assembly strings in C++ tests

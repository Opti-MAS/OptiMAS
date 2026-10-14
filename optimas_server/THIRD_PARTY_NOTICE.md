# Third-Party Notice

## WorkBench

OptiMAS evaluates on the **WorkBench** benchmark, originally created by
[MindsDB](https://mindsdb.com/) and published under the MIT License. It is **not** vendored here; `README.md`
documents cloning it into `optimas_server/workbench/` at setup time.

- **Original repository**: https://github.com/olly-styles/WorkBench
- **Paper**: https://arxiv.org/abs/2405.00823
- **License**: MIT (see `LICENSE` in the cloned `workbench/` directory)
- **Copyright**: (c) 2024 Mindsdb

WorkBench is the first open-source benchmark for evaluating agent performance
on realistic workplace tasks across calendar, email, analytics, project
management, and customer relationship management domains.

Special thanks to Jorge Torres, Adam Carrigan, and the rest of the MindsDB team.

## Adapted WorkBench Evaluation and Data

`optimas_client/src/workbench/official_metrics.py` adapts the state-comparison
metrics from WorkBench. The distributed WorkBench JSONL files derive from the
benchmark's public task data. The upstream MIT notice applies to these materials.
The integration targets commit `42f1b6dc18986f47701ca83de2d63dd4f95c59a0`.

```text
MIT License

Copyright (c) 2024 Mindsdb

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

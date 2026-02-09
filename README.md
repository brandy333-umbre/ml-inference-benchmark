# ml-inference-benchmark
A lightweight ML inference benchmarking harness for measuring latency and throughput of Hugging Face / PyTorch models across batch sizes and sequence lengths.

This project benchmarks inference on already trained models. It measures latency and throughput 
So it will answer questions such as:
- How long does a forward pass take?
- How does latency change with batch size?
- What throughput can the system sustain?
- How costly are longer input sequences?

what model is being benchmarked?
The harness benchmarks any saved Hugging Face
`AutoModelForSequenceClassification` model.

By default, it is used with a **biomedical Transformer text classifier**
trained externally (e.g. on PubMed RCT data) and saved locally
(e.g. `clinical_nlp_model/`).

Why does benchmarking matter?:
Training accuracy alone does not determine whether a model is usable.

In real systems, teams must understand:
- Latency vs throughput trade-offs
- The impact of batching
- The cost of longer inputs
- Hardware-dependent performance

This project focuses on those deployment-relevant questions.

Limitations

Benchmarks a single model at a time
Uses synthetic batching rather than live traffic
Focuses on latency metrics, not memory profiling

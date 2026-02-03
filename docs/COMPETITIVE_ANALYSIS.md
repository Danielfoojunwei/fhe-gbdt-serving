# FHE Machine Learning Inference: Competitive Analysis

**Last Updated:** February 2026
**Document Version:** 1.0

## Executive Summary

This document provides a comprehensive analysis of competitors and existing solutions in the Fully Homomorphic Encryption (FHE) machine learning inference space. The market is experiencing rapid growth, with the global FHE market projected to reach $2.98 billion by 2033, expanding at a CAGR of 24.1%.

---

## Table of Contents

1. [Commercial FHE ML Companies](#1-commercial-fhe-ml-companies)
2. [Open Source FHE Libraries](#2-open-source-fhe-libraries)
3. [Privacy-Preserving ML Platforms](#3-privacy-preserving-ml-platforms)
4. [Comparative Analysis](#4-comparative-analysis)
5. [Market Positioning](#5-market-positioning)

---

## 1. Commercial FHE ML Companies

### 1.1 Zama.ai

**Overview:**
Zama is the world's first FHE unicorn ($1B+ valuation as of June 2025), an open-source cryptography company building state-of-the-art FHE solutions for blockchain and AI. Raised $57 million from Blockchange Ventures and Pantera Capital.

**Products:**
| Product | Description |
|---------|-------------|
| **TFHE-rs** | Pure Rust implementation of TFHE scheme for Boolean and Integer Arithmetics over encrypted data. Core cryptographic library. |
| **Concrete** | Open-source FHE compiler leveraging LLVM that converts Python programs into FHE equivalents. |
| **Concrete ML** | Privacy-preserving ML framework with scikit-learn and PyTorch compatible APIs. |

**Approach to FHE ML:**
- Uses TFHE (Torus FHE) scheme with programmable bootstrapping
- Automatic model conversion from scikit-learn/PyTorch to FHE equivalents
- Quantization Aware Training for low precision (2-3 bits) weights

**Supported ML Models:**
- Tree-based: Decision Trees, Random Forests, XGBoost, LightGBM, CatBoost
- Neural Networks: Fully-connected networks with customizable activations
- Linear Models: Linear/Logistic Regression
- Nearest Neighbors
- LLM fine-tuning (LoRA) on encrypted data

**Performance Benchmarks (TFHE-rs v1.4 on 8xH100 GPUs):**
- Bootstrapping: <1ms for single input (2x speedup vs v1.3)
- Integer logarithm: 3x faster
- Encrypted random generation: 10x faster
- 64-bit integer division: 4x faster on 4+ GPUs
- GPU backend: up to 4.2x faster than CPU

**Deployment Model:**
- Open-source libraries (BSD license)
- Cloud deployment support with client/server architecture
- GPU acceleration support (NVIDIA H100)

**Enterprise Features:**
- Production-ready deployment guides
- Hybrid model compilation
- Serialization support
- Interoperability with TFHE-rs radix ciphertexts

**Pricing Model:**
- Open-source (free)
- Enterprise support available via contact

**Documentation Quality:** Excellent
- Comprehensive docs at docs.zama.ai
- Interactive tutorials and demos
- Community forum with <24hr response time
- Discord community (#concrete channel on fhe.org)

**Community Adoption:** High
- GitHub: concrete-ml ~1k stars, TFHE-rs ~2k stars
- Active development with regular releases (v1.9.0 latest)
- FHE.org community presentations

---

### 1.2 Duality Technologies

**Overview:**
Enterprise-focused FHE company founded by groundbreaking cryptographers. Raised $30M+ in funding. Primary maintainer of OpenFHE open-source library.

**Products:**
| Product | Description |
|---------|-------------|
| **Duality Platform** | Enterprise-ready privacy-enhanced data collaboration platform |
| **OpenFHE** | Open-source FHE library (BSD license) |
| **SecurePlus ML** | ML applications on AWS with Intel acceleration |

**Approach to FHE ML:**
- Uses CKKS scheme optimized for ML applications
- Supports encrypted queries, analytics, and ML train/inference
- LLM fine-tuning and serving with RAG flow support

**Supported ML Models:**
- Linear, Logistic Regression
- XGBoost
- CNN
- LLMs (fine-tuning and inference)
- Custom models via OpenFHE

**Performance:**
- Thousands of encrypted queries per second on lightweight machines
- Private LLM inference framework with CKKS optimizations

**Deployment Model:**
- SaaS platform on AWS/Azure
- On-premises deployment
- Hybrid options
- AWS Nitro Enclaves integration

**Enterprise Features:**
- Microsoft Azure Marketplace integration
- Oracle Cloud Infrastructure support
- Deloitte partnership for enterprise deployment
- DARPA partnership ($14.5M contract)
- Full audit and compliance support

**Notable Customers:**
- NHS England
- National Cancer Institute (NCI)
- UK Department for Science, Innovation and Technology
- Nasdaq (fraud detection)
- Financial services, healthcare organizations

**Pricing Model:**
- Enterprise pricing (contact for quotes)
- Pay-as-you-go options available

**Documentation Quality:** Good
- OpenFHE documentation comprehensive
- Enterprise platform requires sales engagement

**Community Adoption:** High (via OpenFHE)
- OpenFHE: ~1k GitHub stars
- NumFocus fiscally sponsored project
- Academic partnerships (MIT, Intel, Samsung)

**Recognition:**
- Gartner Cool Vendor
- World Economic Forum Tech Pioneer 2021
- Fast Company Most Innovative Companies 2020
- CB Insights AI 100 (2022)

---

### 1.3 Inpher (XOR Secret Computing)

**Overview:**
Founded in 2015, New York-based Secret Computing company. Raised $14M+ (Series A led by JP Morgan). Team includes recognized MPC, FHE, and FL leaders (50% PhDs).

**Products:**
| Product | Description |
|---------|-------------|
| **XOR Secret Computing Platform** | Privacy-preserving ML and analytics platform |
| **XOR Engine** | Core encryption-in-use technology |

**Approach to FHE ML:**
- Hybrid approach: primarily Secure Multiparty Computation (MPC), with FHE for specific use cases
- Data stays localized in separate XOR Machines
- No data transfer or third-party required

**Supported Operations:**
- Privacy-preserving ML training and inference
- Large number of advanced ML operations
- High precision (no noise injection)

**Deployment Model:**
- AWS Marketplace (XOR Secret Computing Platform)
- On-premises deployment
- Hybrid deployment

**Enterprise Features:**
- Data residency compliance
- Cross-jurisdictional collaboration
- No third-party data exposure

**Notable Customers:**
- Global financial services companies
- Healthcare organizations
- Manufacturing companies
- DataCo Technologies partnership

**Use Cases:**
- Financial fraud detection
- Aggregated model features across private datasets
- Heart disease prediction
- IoT applications

**Pricing Model:**
- Enterprise pricing via AWS Marketplace
- Custom enterprise agreements

**Documentation Quality:** Moderate
- AWS Marketplace documentation available
- Technical resources via Inpher website

---

### 1.4 Enveil

**Overview:**
Founded in 2016 by Ellison Anne Williams (former NSA senior researcher). Pioneer in Privacy Enhancing Technology protecting Data in Use. Raised $35M+ (Series B led by USAA).

**Products:**
| Product | Description |
|---------|-------------|
| **ZeroReveal Search** | Encrypted search tool maintaining encryption outside network |
| **ZeroReveal ML** | Machine learning on encrypted data |
| **ZeroReveal ML Encrypted Training (ZMET)** | Encrypted federated learning (launched June 2022) |

**Approach to FHE ML:**
- Homomorphic encryption for Data in Use protection
- Encrypted search, analytics, and ML
- Federated learning on encrypted data

**Deployment Model:**
- On-premises
- Cloud deployment
- Decentralized data collaboration

**Enterprise Features:**
- Cross-organizational/jurisdictional boundary collaboration
- No sensitive variable exposure
- Production-ready and operational

**Investors:**
- USAA
- Mastercard
- Capital One Ventures
- In-Q-Tel (CIA)
- Bloomberg Beta
- C5 Capital

**Recognition:**
- World Economic Forum Technology Pioneer
- Gartner Cool Vendor
- RSA Conference Innovation Sandbox winner (2017)

**Pricing Model:**
- Enterprise pricing (contact required)

**Documentation Quality:** Moderate
- Product-focused documentation
- Technical whitepapers available

---

### 1.5 Cosmian

**Overview:**
French company focused on confidential computing and FHE. Raised $1.5M (Seed). Member of Confidential Computing Consortium.

**Approach to FHE ML:**
- Combination of Functional Encryption, HE, MPC, and Secure Enclaves
- Zero-trust environment with verifiable confidential VM
- 100% encryption with no performance impact claimed

**Products:**
- Confidential AI runner
- KMS (Key Management System)
- AI model deployment, querying, and fine-tuning in confidentiality

**Deployment Model:**
- Cloud-based with confidential VMs
- Hybrid approaches combining FHE with Confidential Computing

**Notable Feature:**
- FHE in secure enclaves to avoid side-channel leakage

---

### 1.6 Google FHE Transpiler / HEIR

**Overview:**
Google's open-source FHE compiler technology for C++ developers.

**Products:**
| Product | Description |
|---------|-------------|
| **FHE C++ Transpiler** | Converts C++ to FHE-equivalent code (exploratory proof-of-concept) |
| **HEIR** | Next-generation FHE compiler toolchain and design environment |

**Approach:**
1. XLS[cc] stage: C++ to XLS IR
2. Optimizer stage: Optimize XLS IR
3. Booleanifier stage: Rewrite to Boolean operations (AND, OR, NOT)
4. FHE IR Transpiler: Translate to FHE-C++
5. FHE testbench: Run with TFHE or OpenFHE's BinFHE

**Supported Backends:**
- TFHE library
- OpenFHE BinFHE library

**Status:**
- Exploratory proof-of-concept
- Run-times likely too long for practical deployment currently
- HEIR planned as migration target for transpiler features

**Repository:** github.com/google/fully-homomorphic-encryption
- ~3.6k GitHub stars
- Active development (last update: August 2025)

**Deployment Model:**
- Open-source library
- Linux only, GCC 9+, Bazel 4.0.0+

---

### 1.7 Microsoft SEAL Ecosystem

**Overview:**
Easy-to-use open-source HE library developed by Microsoft Research Cryptography and Privacy Research Group. Origins from CryptoNets paper.

**Supported Schemes:**
| Scheme | Use Case |
|--------|----------|
| **BFV** | Modular arithmetic on encrypted integers |
| **BGV** | Modular arithmetic on encrypted integers |
| **CKKS** | Additions/multiplications on encrypted real/complex numbers (approximate) |

**ML Applications:**
- Primitive model inference (addition and multiplication)
- CKKS recommended for ML due to real number support
- CNN inference demonstrated

**Supporting Tools:**
- **EVA Compiler**: Helps with CKKS rescaling and scale alignment
- **Private AI Bootcamp**: Training resources

**Performance:**
- SEAL achieves shortest latency (<56% of OpenFHE, <17% of HElib in some tests)
- CKKS fastest at 0.031ms per operation

**Deployment Model:**
- Open-source library (MIT license)
- Cross-platform (Windows, Linux, macOS)
- Wrappers available (Python, .NET)

**Community Adoption:** Highest
- ~3.9k GitHub stars
- Most widely used FHE library
- Academic standard for benchmarks

**Documentation Quality:** Excellent
- Comprehensive documentation
- Microsoft Research tutorials and bootcamps
- Academic papers and examples

---

### 1.8 Intel HE-Toolkit

**Overview:**
Intel's homomorphic encryption toolkit optimized for Intel platforms.

**Components:**
| Component | Description |
|-----------|-------------|
| **Intel HE Toolkit** | Sample kernels and examples using SEAL, PALISADE, HElib |
| **Intel HEXL** | CPU-based acceleration library (up to 7.2x speedup) |
| **nGraph-HE** | Deep learning with HE through Intel nGraph |

**Supported Libraries:**
- Microsoft SEAL
- PALISADE (now OpenFHE)
- HElib

**ML Applications:**
- Transposed logistic regression inference
- Neural network inference (nGraph-HE, nGraph-HE2)
- SEAL CKKS scheme for ML

**Performance:**
- Intel HEXL: up to 7.2x speedup vs native C++
- Optimized for 3rd Gen Intel Xeon Scalable processors
- Built-in AI accelerators

**Notable Deployments:**
- Nasdaq: HE for fraud/money laundering detection with AI/ML
- Duality SecurePlus ML on AWS with Intel Xeon

**Repository:** github.com/IntelLabs/he-toolkit

---

## 2. Open Source FHE Libraries

### 2.1 OpenFHE

**Overview:**
Successor to PALISADE, comprehensive open-source FHE library maintained by Duality with community contributions. Version 1.4.2 (October 2025).

**Supported Schemes:**
- BFV (integer arithmetic)
- BGV (integer arithmetic)
- CKKS (approximate real/complex arithmetic with bootstrapping)
- DM/CGGI/LMKCDEY schemes

**Key Features:**
- All common FHE schemes implemented
- Bootstrapping support
- Multi-party computation protocols
- Hardware acceleration support

**Performance (2025 Benchmarks):**
| Scheme | Performance |
|--------|-------------|
| BFV | 0.055ms per operation (fastest in benchmarks) |
| BGV | Competitive |
| CKKS | Division support (unique feature) |

**GPU Acceleration:**
- FIDESlib: First open-source server-side CKKS GPU library
- Fully interoperable with client-side OpenFHE
- Up to 74x speedup over alternatives for bootstrapping

**Community:**
- ~1k GitHub stars
- NumFocus fiscally sponsored
- Active academic community

**License:** BSD 2-clause

---

### 2.2 Microsoft SEAL

**Overview:**
Most popular FHE library by GitHub stars. Production-quality implementation.

**Supported Schemes:**
- BFV
- BGV
- CKKS

**Performance (2025 Benchmarks):**
| Operation | Latency |
|-----------|---------|
| CKKS add/sub | 0.031ms (fastest) |
| Overall | 50% faster than OpenFHE for CNN |

**Community:**
- ~3.9k GitHub stars
- Most academic citations
- Wide language bindings

**License:** MIT

---

### 2.3 TFHE-rs (Zama)

**Overview:**
Pure Rust implementation of TFHE scheme. Modern, memory-safe implementation.

**Key Features:**
- Boolean and integer arithmetic
- Programmable bootstrapping
- GPU acceleration (CUDA)
- HPU (Homomorphic Processing Unit) support

**Performance (v1.4, 8xH100 GPUs):**
| Operation | Performance |
|-----------|-------------|
| Bootstrapping | <1ms single input |
| GPU vs CPU | 4.2x faster |
| Multi-GPU multiplication | 3x faster |

**Community:**
- ~2k GitHub stars
- Active development
- Regular releases

**License:** BSD-3-Clause

---

### 2.4 HElib

**Overview:**
Oldest FHE library (first release: May 2013). Developed by Shai Halevi and Victor Shoup at IBM.

**Supported Schemes:**
- BGV (with bootstrapping)
- CKKS

**Key Features:**
- SIMD operations
- Smart-Vercauteren ciphertext packing
- Gentry-Halevi-Smart optimizations
- Multi-threading support

**Performance:**
- BGV add/sub: 0.021ms (fastest in benchmarks)
- Slower initialization and decryption (800x slower decryption than SEAL)

**Higher-Level Tools:**
- IBM HElayers SDK: High-level API for data scientists
- Supports linear regression, logistic regression, neural networks
- Python API available

**Community:**
- ~3.2k GitHub stars
- Academic standard
- IBM support

**License:** Apache 2.0

---

### 2.5 Lattigo

**Overview:**
Go implementation of lattice-based multiparty HE. Originally developed at EPFL, now maintained by Tune Insight SA.

**Supported Schemes:**
- BFV
- BGV
- CKKS (with bootstrapping)
- RGSW

**Unique Features:**
- Pure Go implementation
- Cross-platform including WASM for browser
- Multiparty HE protocols (threshold/distributed)
- Natural concurrency model

**Pre-built Circuits:**
- Polynomial evaluation
- Minimax approximation
- Comparison operations
- Inverse computation
- Mod1 function
- DFT (Discrete Fourier Transform)

**Community:**
- Acknowledged by HomomorphicEncryption.org
- Active development (v5 available)

**License:** Apache 2.0

---

## 3. Privacy-Preserving ML Platforms

### 3.1 TensorFlow Federated (TFF)

**Overview:**
Google's open-source framework for federated learning with privacy preservation.

**Key Features:**
| Feature | Description |
|---------|-------------|
| Federated Learning API | Apply existing ML models to decentralized data |
| Federated Core API | Lower-level control for custom algorithms |
| Differential Privacy | User-level DP via modified DP-SGD |
| Secure Aggregation | Cryptographic aggregation protocols |

**Privacy Techniques:**
- Differential Privacy (DP)
- Secure Aggregation
- Can combine with HE

**Use Cases:**
- Mobile keyboard predictions
- Healthcare data analysis
- Cross-organization model training

**Documentation Quality:** Excellent
- Comprehensive tutorials
- Colab notebooks
- Academic paper support

---

### 3.2 PySyft (OpenMined)

**Overview:**
Open-source library for privacy-preserving ML developed by OpenMined community.

**Key Features:**
- Federated Learning
- Differential Privacy
- Secure Multi-Party Computation (MPC)
- Homomorphic Encryption integration

**Supported Frameworks:**
- PyTorch (primary)
- TensorFlow
- Any Python code with third-party libraries

**Concept: Remote Data Science**
- "Bring code to data, not data to code"
- Datasites enable queries without seeing data
- Data owner controls acceptable use

**Deployment:**
- Linux, macOS, Windows
- Docker, Kubernetes
- Cloud deployment support

**Use Cases:**
- Healthcare (multi-hospital collaboration)
- Finance (fraud detection without data sharing)
- Retail (cross-store recommendations)
- Automotive (fleet sensor data)

**Community:**
- PyTorch partnership for PPML development
- Active Discord community
- Fellowship funding program

---

### 3.3 CrypTen (Meta/Facebook)

**Overview:**
Privacy-preserving ML framework built on PyTorch using Secure Multi-Party Computation.

**Key Features:**
- PyTorch-like API
- Automatic differentiation
- Modular neural networks in MPC
- Native GPU support
- Arbitrary number of parties

**Performance:**
- 2.5-3 orders of magnitude slower than plain PyTorch
- Faster than real-time for speech recognition (Wav2Letter)

**Threat Model:**
- Semi-honest security
- Multiple parties supported

**Use Cases:**
- Medical research collaboration
- Salary data analysis
- Any multi-party ML scenario

**Repository:** github.com/facebookresearch/CrypTen
- Published at NeurIPS 2021

---

### 3.4 MP-SPDZ

**Overview:**
Versatile framework for benchmarking 30+ MPC protocol variants. Developed by CSIRO's Data61.

**Supported Protocols:**
| Category | Protocols |
|----------|-----------|
| Secret Sharing | SPDZ, SPDZ2k, MASCOT, Overdrive |
| Garbled Circuits | BMR, Yao's |
| Replicated SS | Three-party, Shamir's |

**Security Models:**
- Honest majority
- Dishonest majority
- Semi-honest (passive)
- Malicious (active)

**Circuit Types:**
- Binary circuits
- Arithmetic circuits (modulo primes)
- Arithmetic circuits (powers of two)

**Key Features:**
- High-level Python interface
- Comprehensive protocol coverage
- Academic benchmark standard

**Repository:** github.com/data61/MP-SPDZ

---

### 3.5 ABY Framework

**Overview:**
Framework for efficient mixed-protocol secure two-party computation. Presented at NDSS 2015.

**Sharing Types:**
| Type | Protocol Base |
|------|---------------|
| Arithmetic (A) | Beaver's multiplication triples |
| Boolean (B) | GMW protocol |
| Yao (Y) | Yao's garbled circuits |

**Key Innovation:**
- Efficient conversion between sharing types
- Pre-computation of cryptographic operations
- Oblivious transfer extensions

**Performance (ABY2.0):**
- Online throughput: 496-754x improvement over ABY
- 2x improvement in online communication
- Dimension-independent dot product protocol

**Extensions:**
- ABY3/ABY4: Three/four party honest-majority
- HyCC: Automatic C partitioning compiler

**Repository:** github.com/encryptogroup/ABY

---

## 4. Comparative Analysis

### 4.1 FHE Library Comparison

| Library | Schemes | GitHub Stars | License | Primary Language | Best For |
|---------|---------|--------------|---------|------------------|----------|
| Microsoft SEAL | BFV, BGV, CKKS | ~3.9k | MIT | C++ | General HE, CKKS performance |
| HElib | BGV, CKKS | ~3.2k | Apache 2.0 | C++ | BGV operations, IBM ecosystem |
| OpenFHE | BFV, BGV, CKKS, TFHE | ~1k | BSD | C++ | Comprehensive scheme support |
| TFHE-rs | TFHE | ~2k | BSD-3 | Rust | Boolean/Integer, GPU accel |
| Lattigo | BFV, BGV, CKKS | ~1k | Apache 2.0 | Go | Multiparty HE, Go ecosystem |

### 4.2 Performance Benchmarks (2025)

| Operation | SEAL | HElib | OpenFHE | TFHE-rs |
|-----------|------|-------|---------|---------|
| CKKS Add/Sub | 0.031ms | 0.063ms | 0.047ms | N/A |
| BGV Add/Sub | 0.04ms | 0.021ms | 0.045ms | N/A |
| BFV Multiply | Slower | N/A | 0.055ms | N/A |
| Bootstrapping | N/A | N/A | Supported | <1ms (GPU) |

### 4.3 Commercial Platform Comparison

| Company | Primary Tech | Deployment | ML Support | Target Market |
|---------|--------------|------------|------------|---------------|
| Zama | TFHE | OSS + Enterprise | Excellent | Developers, Enterprise |
| Duality | CKKS/OpenFHE | SaaS/On-prem | Good | Enterprise |
| Inpher | MPC + FHE | AWS/On-prem | Good | Enterprise |
| Enveil | HE | On-prem/Cloud | Moderate | Government, Finance |
| Cosmian | FHE + CC | Cloud | Moderate | Enterprise |

### 4.4 Privacy-Preserving ML Platform Comparison

| Platform | Primary Technique | Framework Support | GPU | Maturity |
|----------|-------------------|-------------------|-----|----------|
| TensorFlow Federated | FL + DP | TensorFlow | Yes | Production |
| PySyft | FL + MPC + DP | PyTorch, TF | Yes | Production |
| CrypTen | MPC | PyTorch | Yes | Research/Production |
| MP-SPDZ | MPC (30+ protocols) | Python | No | Research |
| ABY | Mixed MPC | C++ | No | Research |

---

## 5. Market Positioning

### 5.1 Our Differentiation (FHE-GBDT-Serving)

Based on the competitive analysis, our FHE-GBDT-Serving platform has unique positioning:

| Aspect | Our Approach | Competition |
|--------|--------------|-------------|
| **Focus** | GBDT-specific optimization | General-purpose FHE ML |
| **Models** | XGBoost, LightGBM, CatBoost | Various (often generic tree support) |
| **Deployment** | Production-ready serving | Often library/SDK focused |
| **Performance** | Optimized for tree inference | General optimization |

### 5.2 Competitive Advantages

1. **Specialized Tree Optimization**: While Zama's Concrete-ML supports trees, our focus on GBDT-specific optimizations may provide performance advantages for this use case.

2. **Production Serving Focus**: Most competitors provide libraries/SDKs; we provide a complete serving infrastructure.

3. **Framework Agnostic**: Support for multiple GBDT frameworks (XGBoost, LightGBM, CatBoost) with unified API.

4. **Enterprise-Ready**: Compliance audits, security certifications, operational documentation.

### 5.3 Competitive Threats

1. **Zama Concrete-ML**: Most direct competitor with excellent documentation, community, and $1B valuation funding.

2. **Duality Platform**: Strong enterprise positioning and partnerships (Microsoft, Intel, DARPA).

3. **General-purpose improvement**: As FHE performance improves, specialized optimizations may become less critical.

### 5.4 Market Opportunity

- FHE market growing at 24.1% CAGR to $2.98B by 2033
- Large enterprises represent 70.1% of market (2024)
- SME segment fastest growing (11.2% CAGR)
- Cloud-based FHE solutions driving adoption

---

## Sources

### Commercial Companies
- [Zama GitHub](https://github.com/zama-ai)
- [Zama Concrete-ML Documentation](https://docs.zama.org/concrete-ml)
- [TFHE-rs Benchmarks](https://docs.zama.org/tfhe-rs/get-started/benchmarks)
- [Duality Technologies](https://dualitytech.com/)
- [Duality OpenFHE](https://dualitytech.com/platform/open-source/)
- [Inpher](https://inpher.io/)
- [Inpher on AWS](https://aws.amazon.com/blogs/awsmarketplace/using-the-xor-secret-computing-platform-for-machine-learning-on-private-data-sources/)
- [Enveil](https://www.enveil.com/)
- [Cosmian](https://cosmian.com/)
- [Google FHE Transpiler](https://github.com/google/fully-homomorphic-encryption)
- [Microsoft SEAL](https://github.com/microsoft/SEAL)
- [Intel HE-Toolkit](https://github.com/IntelLabs/he-toolkit)

### Open Source Libraries
- [OpenFHE](https://openfhe.org/)
- [OpenFHE GitHub](https://github.com/openfheorg/openfhe-development)
- [HElib GitHub](https://github.com/homenc/HElib)
- [Lattigo GitHub](https://github.com/tuneinsight/lattigo)
- [TFHE-rs Documentation](https://docs.zama.org/tfhe-rs)

### Privacy-Preserving ML
- [TensorFlow Federated](https://www.tensorflow.org/federated)
- [PySyft GitHub](https://github.com/OpenMined/PySyft)
- [CrypTen GitHub](https://github.com/facebookresearch/CrypTen)
- [MP-SPDZ GitHub](https://github.com/data61/MP-SPDZ)
- [ABY Framework GitHub](https://github.com/encryptogroup/ABY)

### Research & Benchmarks
- [ACM 2025 HE Library Benchmark Study](https://dl.acm.org/doi/10.1145/3729706.3729711)
- [FHE.org Resources](https://fhe.org/resources/)
- [HomomorphicEncryption.org](https://homomorphicencryption.org/)

### Market Research
- [Homomorphic Encryption Services Market Report](https://researchintelo.com/report/homomorphic-encryption-services-market)
- [FHE Market Analysis](https://www.databridgemarketresearch.com/reports/global-fully-homomorphic-encryption-market)
